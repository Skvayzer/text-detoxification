import os
import sys
import pandas as pd
from tqdm import tqdm

from dotenv import find_dotenv, load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
root = os.path.dirname(find_dotenv())
sys.path.append(root)


def test(
    model,
    test_data,
    tokenizer,
    batch_size=100,
):
    """
    Test function for T5 model: adds prefix to source and trim sequence.
    Then feed to the model and decode the output
    """

    prefix = "make sentence non-toxic:"
    toxic = "source"
    res = pd.DataFrame({"source": test_data[toxic]})
    model_res = []

    for i in tqdm(range(0, len(test_data), batch_size)):
        batch = test_data[i : i + batch_size]
        input_texts = [prefix + line for line in batch[toxic]]

        input_ids = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).input_ids
        outputs = model.generate(input_ids=input_ids)

        decoded_outputs = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        model_res.extend(decoded_outputs)

    res["target"] = model_res
    return res


def predict(model_name=None, model_base="t5-small", dataset=None):
    """Function to get a list of predictions from the model"""

    if dataset is None:
        data_dir = os.getenv("DATA_DIR")
        dataset = root + "/" + data_dir + "/interim/dataset/test.csv"
        dataset = pd.read_csv(dataset, index_col=0)

    if model_name is None:
        model_dir = os.getenv("MODEL_DIR")
        model_name = root + "/" + os.path.join(model_dir, "t5-small-ft")

    # loading the model and run inference for it
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_base)

    res = test(model, dataset, tokenizer)

    return res