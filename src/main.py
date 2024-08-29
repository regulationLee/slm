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

    diagnosis_str = load_medical_data_poc()

    print("\n")
    print("=" * 50)
    print("Read Report Template and Apply User Info ")
    print("=" * 50)
    report_file = '손해사정_보고서_샘플.docx'
    file_path = os.path.join(DATA_PATH, report_file)
    start_time = time.time()
    document = read_doc_report(file_path)
    doc_time = (time.time() - start_time) / 60

    print(f'\nDocument Process time: {doc_time:.2f}s')

    doc_prompt = '\n\n 위의 손해사정보고서의 스타일 분석'
    input_prompt = doc_prompt

    start_time = time.time()
    result_stream = llm_inference(document[0].text + input_prompt, stream=True)
    llm_doc_time = (time.time() - start_time) / 60

    print(f'SLM Document Process time: {llm_doc_time:.2f}s')

    contents_output = ''
    for chunk in result_stream:
        content = chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
        contents_output += content
    print('\n')

    ############## version00 #############3

    # final_prompt = f'\n\n {report_format_output} 위의 손해사정보고서의 내용을 다음의 내용을 바탕으로 수정해줘'
    # final_stream = llm_inference(final_prompt + diagnosis_str, stream=True)
    print("=" * 50)
    # final_prompt = '위의 손해사정보고서의 스타일로 아래의 조사기록을 분석해서 보험금 지급 여부 판단을 포함한 손해사정보고서 작성'
    final_prompt = '위의 손해사정보고서의 스타일로 아래의 조사기록을 분석해서 보험금 지급 여부 판단을 포함한 손해사정보고서 작성'
    final_stream = llm_inference(final_prompt + user_info_prompt + diagnosis_str, stream=True)

    contents_output = ''
    for chunk in final_stream:
        content = chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
        contents_output += content
    print('\n')

    print("=" * 50)
    final_prompt = f'{contents_output} \n\n 위의 손해사정보고서 내용에서 질병 또는 부상정보별로 보험금 지급 여부 판단.'
    final_stream = llm_inference(final_prompt, stream=True)

    for chunk in final_stream:
        print(chunk['message']['content'], end='', flush=True)
    print('\n')

    print("=" * 50)
    final_prompt = f'{contents_output} \n\n 위의 손해사정보고서 내용을 일자별로 개조식으로 요약.'
    final_stream = llm_inference(final_prompt, stream=True)

    for chunk in final_stream:
        print(chunk['message']['content'], end='', flush=True)
    print('\n')



    # ############# version 01 ##################
    # # final_prompt = f'\n\n {report_format_output} 위의 손해사정보고서의 내용을 다음의 내용을 바탕으로 수정해줘'
    # # final_stream = llm_inference(final_prompt + diagnosis_str, stream=True)
    # print("=" * 50)
    # # final_prompt = '위의 손해사정보고서의 스타일로 아래의 조사기록을 분석해서 보험금 지급 여부 판단을 포함한 손해사정보고서 작성'
    # final_prompt = '위의 손해사정보고서의 스타일로 아래의 조사기록을 분석해서 손해사정보고서 작성'
    # final_stream = llm_inference(final_prompt + user_info_prompt + diagnosis_str, stream=True)
    #
    # contents_output = ''
    # for chunk in final_stream:
    #     content = chunk['message']['content']
    #     print(chunk['message']['content'], end='', flush=True)
    #     contents_output += content
    # print('\n')
    #
    # print("=" * 50)
    # final_prompt = f'{contents_output} \n\n 위의 손해사정보고서 내용을 기반으로 보험금 지급 여부 판단.'
    # final_stream = llm_inference(final_prompt, stream=True)
    #
    # contents_output = ''
    # for chunk in final_stream:
    #     content = chunk['message']['content']
    #     print(chunk['message']['content'], end='', flush=True)
    #     contents_output += content
    # print('\n')

print('Done')

