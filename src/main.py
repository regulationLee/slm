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

from text_parser import *
from poc_medical_record import *

DATA_PATH = '/Users/glee/Desktop/sakak/TSA/samples/'
RESULT_PATH = '/Users/glee/Desktop/sakak/slm/results/'


def llm_inference(prompt, stream):
    stream = ollama.chat(
      model='llama3.1:latest',
      messages=[{'role': 'user', 'content': prompt}],
      stream=stream
    )
    return stream


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--task", type=str, default="ocr", help='doc or ocr')
    parser.add_argument('-c', "--convert_tiff", action='store_true')
    parser.add_argument('-cr', "--tiff_ratio", type=float, default=0.25)
    parser.add_argument('-slm', "--gen_report", action='store_true')

    parser.add_argument('-r', "--readtype", type=str, default="batch", help='batch or page')
    parser.add_argument('-nn', "--name", type=str, default="성춘향")
    parser.add_argument('-na', "--age", type=str, default="38")
    parser.add_argument('-nsd', "--start", type=str, default="2024-06-01")
    parser.add_argument('-ned', "--end", type=str, default="2024-06-10")
    parser.add_argument('-npoc', "--poc", type=str, default="이초랑")
    parser.add_argument('-ninc', "--inc", type=str, default="TSA 손해사정")
    parser.add_argument('-nsv', "--supervisor", type=str, default="김바다")
    parser.add_argument('-nl', "--leader", type=str, default="이칠성")
    parser.add_argument('-ncon', "--contract", type=str, default="가나다 건강보험")
    parser.add_argument('-nconno', "--connum", type=str, default="0000048번")
    parser.add_argument('-ncond', "--conday", type=str, default="2021-07-01")
    parser.add_argument('-ncons', "--constatus", type=str, default="정상유지")

    args = parser.parse_args()
    args.data_path = DATA_PATH
    args.result_path = RESULT_PATH
    user_info = parse_input(args)
    user_info_prompt = ', '.join(str(value) for value in user_info.values())

    args.gen_report = True
    # args.convert_tiff = True

    # diagnosis_str = load_medical_data_poc()

    # print("\n")
    # print("=" * 50)
    # print("Read Report Template and Apply User Info ")
    # print("=" * 50)
    # report_file = '손해사정_보고서_샘플.docx'
    # file_path = os.path.join(DATA_PATH, report_file)
    # start_time = time.time()
    # document = read_report(file_path)
    # doc_time = (time.time() - start_time) / 60
    #
    # print(f'\nDocument Process time: {doc_time:.2f}s')
    #
    # doc_prompt = '\n\n 위의 손해사정보고서의 내용에서 다음의 정보만 수정한 새로운 손해사정보고서를 만들어줘'
    # input_prompt = doc_prompt + '  ' + user_info_prompt
    #
    # start_time = time.time()
    # result_stream = llm_inference(document[0].text + input_prompt, stream=True)
    # llm_doc_time = (time.time() - start_time) / 60
    #
    # print(f'SLM Document Process time: {llm_doc_time:.2f}s')
    #
    # for chunk in result_stream:
    #     print(chunk['message']['content'], end='', flush=True)
    # print('\n')

    # format_prompt = result_stream['message']['content']

    # final_prompt = '\n\n 아래의 소견서 내용이 한 명의 환자에 대한 소견서일 때 아래의 내용을 한 문장으로 자세하게 요약해줘'
    # final_stream = llm_inference(final_prompt + diagnosis_str, stream=True)
    #
    # for chunk in final_stream:
    #     print(chunk['message']['content'], end='', flush=True)
    # print('\n')

    final_prompt = '파이토치로 Cifar-10 classification을 학습하는 코드 예제'
    final_stream = llm_inference(final_prompt, stream=True)

    output = ''
    for chunk in final_stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        output += content
    print('\n')

    final_prompt = f'{output} 위의 코드에서 사용한 옵티마이저에 대해 설명'
    final_stream = llm_inference(final_prompt, stream=True)

    for chunk in final_stream:
        print(chunk['message']['content'], end='', flush=True)
    print('\n')

print('Done')