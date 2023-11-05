import os
import sys
import warnings
import datasets
import numpy as np
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, load_metric, concatenate_datasets

warnings.filterwarnings("ignore")


def preprocess_function(examples, tokenizer):
    """Preprocess function for T5 model: adds prefix to source and trim sequence"""
    prefix = "make sentence non-toxic:"

    max_input_length = 256
    max_target_length = 256
    toxic = "source"
    non_toxic = "target"

    inputs = [prefix + ex if ex else " " for ex in examples[toxic]]
    targets = [ex if ex else " " for ex in examples[non_toxic]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_df(data: pd.DataFrame,
                  toxicity_threshold=0.99
                  ):


    mask = data["trn_tox"] > data["ref_tox"]
    temp = data.loc[mask, "reference"].copy()
    data.loc[mask, "reference"] = data.loc[mask, "translation"]
    data.loc[mask, "translation"] = temp


    filtered_data = data[
    ((data["ref_tox"] > toxicity_threshold) & (data["trn_tox"] < 1 - toxicity_threshold))
    | ((data["trn_tox"] > toxicity_threshold) & (data["ref_tox"] < 1 - toxicity_threshold))
    ]
    # Preprocess entries for 'reference' and 'translation' columns
    data_preprocessed = filtered_data.copy()
    data_preprocessed['reference'] = data_preprocessed['reference'].apply(remove_symbols)
    data_preprocessed['translation'] = data_preprocessed['translation'].apply(remove_symbols)

    return data_preprocessed



# compute metrics function to pass to trainer
def compute_metrics(eval_preds, tokenizer, metric):
    """Compute metrics for model evaluation"""

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import transformers
from sklearn.model_selection import train_test_split




# Create a custom dataset class
class TextDetoxDataset(Dataset):
    def __init__(self, tokenizer, data_df, max_length=512):
        self.tokenizer = tokenizer
        self.input_texts = data_df['reference']
        self.target_texts = data_df['translation']
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, index):
        # Tokenize the input and target text
        input_text = self.input_texts.iloc[index]
        target_text = self.target_texts.iloc[index]

        source = self.tokenizer.__call__(input_text, max_length=self.max_length, truncation=True)

        target = self.tokenizer.__call__(target_text, max_length=self.max_length, truncation=True)

        return {
            'input_ids': source['input_ids'],
            'attention_mask': source['attention_mask'],
            'labels': target['input_ids']
        }


def train(
    checkpoint="t5-small",
    random_state=42,
    dataset=None,
    metric="sacrebleu",
    tokenizer=None,
):
    """Train the model with given checkpoint on the dataset and save the results"""

    data = preprocess_df(dataset)

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(data, test_size=0.1, random_state=42)

    # Select only the necessary columns for training and validation
    train_df = train_df[['reference', 'translation']]
    val_df = val_df[['reference', 'translation']]

    # Create the training and validation datasets
    train_dataset = TextDetoxDataset(tokenizer, train_df)
    val_dataset = TextDetoxDataset(tokenizer, val_df)
    

    metric = load_metric(metric)

    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # defining the parameters for training
    batch_size = 32
    args = Seq2SeqTrainingArguments(
        f"{checkpoint}-ft",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # instead of writing train loop we will use Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric),
    )

    trainer.train()
    # decide if ft2 or ft
    trainer.save_model(
        checkpoint
    )