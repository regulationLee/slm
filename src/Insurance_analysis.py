# -*- coding: utf-8 -*-
# !/usr/bin/env python
import os
import io
import numpy
import fitz
import time
import argparse
import ollama
import pandas as pd
import numpy as np
import platform

from text_parser import *
from poc_medical_record import *

if platform.system() == 'Linux':
    DATA_PATH = '/home/glee/sakak/data/TSA/samples/'
    RESULT_PATH = '/home/glee/sakak/slm/results/'
else:
    DATA_PATH = '/Users/glee/Desktop/sakak/TSA/samples/'
    RESULT_PATH = '/Users/glee/Desktop/sakak/slm/results/'


def llm_inference(prompt, stream):
    if platform.system() == 'Linux':
        model_name = 'aya:35b'
    else:
        model_name = 'aya:8b'

    stream = ollama.chat(
      model=model_name,
      messages=[{'role': 'user', 'content': prompt}],
      stream=stream
    )
    return stream


if __name__ == "__main__":
    diagnosis_str = load_medical_data_poc()

    record_file = '상해_상품안내서.pdf'
    file_path = os.path.join(DATA_PATH, record_file)
    contents = read_pdf_files(file_path)

    doc_prompt = f' {contents} \n\n 위의 보험약관에서 다음의 의료기록에 해당하는 내용을 찾아줘 \n\n {diagnosis_str}'
    # doc_prompt = f'다음의 약관내용을 분석해줘 {contents}'
    input_prompt = doc_prompt

    start_time = time.time()
    result_stream = llm_inference(contents + input_prompt, stream=True)
    llm_doc_time = (time.time() - start_time) / 60

    print(f'SLM Document Process time: {llm_doc_time:.2f}s')

    contents_output = ''
    for chunk in result_stream:
        content = chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
        contents_output += content
    print('\n')
