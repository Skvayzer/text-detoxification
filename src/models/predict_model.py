import os
import sys
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers


ModelTokenizer = transformers.models.t5.tokenization_t5_fast.T5TokenizerFast
ModelType = transformers.models.t5.modeling_t5.T5ForConditionalGeneration

def inference(text_model: ModelType, inference_request: str, tokenizer: ModelTokenizer) -> str:
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = text_model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

def predict(model_cktp_path="t5-small", text="Fuck it I'm done with this shit goddamn"):
    """Function to get a list of predictions from the model"""

    # loading the model and run inference for it
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_cktp_path)
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_cktp_path)

    res = inference(model, text, tokenizer)

    return res