# -*- coding: utf-8 -*-
## local model directory: .cache/huggingface/hub
import time
import argparse
import ollama
import pandas as pd
import numpy as np
import platform
import itertools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, pipeline
from huggingface_hub import HfFolder

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import pdfplumber


HfFolder.save_token("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")
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=False)

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.bfloat16,
    device_map="mps",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])

##### transformer 활용 기본형

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)

messages_v2 = "You are a pirate chatbot who always responds in pirate speak!, Who are you?"

encoded = tokenizer(messages_v2, return_tensors="pt")
model_inputs = encoded.to("mps")
model.to("mps")

generation_config = {
            "max_length": 100,
            "num_beams": 4,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.eos_token_id
        }

generated_ids = model.generate(**model_inputs,  **generation_config)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


#####




# tokenizer = AutoTokenizer.from_pretrained("facebook/MobileLLM-125M", use_fast=False)
# model = AutoModelForCausalLM.from_pretrained("facebook/MobileLLM-125M", trust_remote_code=True)
#
# tokenizer.add_special_tokens(
#     {
#         "eos_token": "</s>",
#         "bos_token": "<s>",
#         "unk_token": "<unk>",
#     }
# )
#
# messages_v2 = "Who are you?"
#
# encoded = tokenizer(messages_v2, return_tensors="pt")
# model_inputs = encoded.to("mps")
# model.to("mps")
#
# generation_config = {
#             "max_length": 100,
#             "num_beams": 4,
#             "no_repeat_ngram_size": 2,
#             "early_stopping": True,
#             "do_sample": True,
#             "temperature": 0.7,
#             "pad_token_id": model.config.eos_token_id
#         }
#
# # generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# generated_ids = model.generate(**model_inputs,  **generation_config)
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0])