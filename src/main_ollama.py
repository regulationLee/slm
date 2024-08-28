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

    if args.task == "doc":
        ####### read report sample and convert to str ############
        print("\n")
        print("=" * 50)
        print("Read Report Template and Apply User Info ")
        print("=" * 50)
        report_file = '손해사정_보고서_샘플.docx'
        file_path = os.path.join(DATA_PATH, report_file)
        start_time = time.time()
        document = read_report(file_path)
        doc_time = (time.time() - start_time) / 60

        print(f'\nDocument Process time: {doc_time:.2f}s')

        doc_prompt = '\n\n 위의 손해사정보고서의 내용을 바탕으로 다음의 정보를 이용하여 새로운 손해사정보고서를 만들어줘'
        input_prompt = doc_prompt + '  ' + user_info_prompt

        start_time = time.time()
        result_stream = llm_inference(document[0].text + input_prompt, stream=True)
        llm_doc_time = (time.time() - start_time) / 60

        print(f'SLM Document Process time: {llm_doc_time:.2f}s')

        for chunk in result_stream:
            print(chunk['message']['content'], end='', flush=True)
        print('\n')

    elif args.task == 'ocr':
        ####### read medical report and convert to str ############
        pages_to_read = 1

        record_file = '차트정보_w_ocr.pdf'
        file_path = os.path.join(DATA_PATH, record_file)
        pdf_document = fitz.open(file_path)
        num_pages = pdf_document.page_count
        extracted_text = []

        record_list = []
        words_by_line = []
        start_time = time.time()

        # save pdf to tiff
        if args.convert_tiff:
            pdf_to_tiff(args, record_file)

        if args.readtype == 'batch':
            print("\n")
            print("=" * 50)
            print("Read Medical Records in a Single batch")
            print("=" * 50)

            file_path = 'ocr_result_words_batch.txt'

            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    words_by_line = file.read()
            else:
                ## read pdf by batch
                result_text = read_records(file_path)
                result_lines = result_text.split('\n')

                for line in result_lines:
                    # pattern = r'\b[가-힣a-zA-Z0-9.-:]+\b'
                    pattern = r'[가-힣a-zA-Z0-9.-:]+'
                    result_words = re.findall(pattern, line)
                    words_by_line.append(' '.join(result_words))

                with open(file_path, 'w', encoding='utf-8') as file:
                  for line in words_by_line:
                    file.write(f"{line}\n")

            combined_result = words_by_line

        elif args.readtype == 'page':
            print("\n")
            print("=" * 50)
            print("Read Medical Records in Batches by Page")
            print("=" * 50)
            ## read pdf by page
            file_path = "ocr_result_words_page.txt"

            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    combined_result = file.read()
            else:
                for page_num in range(0, num_pages, pages_to_read):
                    print(f'Processing pages: {page_num + 1} to {min(page_num + pages_to_read, num_pages)}')

                    patient_logs_list = []
                    words_by_line = []
                    for i in range(pages_to_read):
                      if page_num + i < num_pages:
                        page = pdf_document[page_num + i]
                        patient_logs = read_records_by_page(page, easyocr_cond=False)
                        result_lines = patient_logs.split('\n')
                        for line in result_lines:
                          pattern = r'\b[가-힣a-zA-Z0-9.-:]+\b'
                          # pattern = r'[가-힣a-zA-Z0-9.-:]+'
                          result_words = re.findall(pattern, line)
                          if result_words:
                            words_by_line.append(' '.join(result_words))

                        # patient_logs_list.append(words_by_line)

                    combined_patient_logs = '\n'.join(words_by_line)

                    record_prompt = '\n\n 위의 내용에서 사고내용, 증상, 의사 소견에 해당하는 내용을 보여줘'
                    record_stream = llm_inference(combined_patient_logs + record_prompt, stream=False)

                    record_list.append(record_stream['message']['content'])

                combined_result = '\n'.join(record_list)

                with open(file_path, "w", encoding='utf-8') as text_file:
                    text_file.write(combined_result)

                ocr_time = (time.time() - start_time) / 60
                print(f'Document Process time: {ocr_time:.2f}s')

    elif args.task == 'ocr_test':
        record_file = '상해_상품안내서.pdf'
        file_path = os.path.join(DATA_PATH, record_file)
        pdf_document = fitz.open(file_path)
        num_pages = pdf_document.page_count
        extracted_text = []

        record_list = []
        words_by_line = []
        start_time = time.time()

        print("\n")
        print("=" * 50)
        print("Read Test Sample using OCR")
        print("=" * 50)

        result_text = read_records(file_path)
        result_lines = result_text.split('\n')

        for line in result_lines:
            # pattern = r'\b[가-힣a-zA-Z0-9.-:]+\b'
            pattern = r'[가-힣a-zA-Z0-9.-:]+'
            result_words = re.findall(pattern, line)
            words_by_line.append(' '.join(result_words))

        file_path = 'OCR_TEST.txt'

        with open(file_path, 'w', encoding='utf-8') as file:
            for line in words_by_line:
                file.write(f"{line}\n")

        combined_result = words_by_line

    print(f'Text length of Medical Records: {len(combined_result)}')

    if args.gen_report:
        print("\n")
        print("=" * 50)
        print("Generate Insurance Assessment Report Assistant")
        print("=" * 50)
        lines = combined_result.split('\n')

        df = pd.DataFrame(lines, columns=['Text'])
        df.replace('', np.nan, inplace=True)
        df_cleaned = df.dropna()
        df_cleaned = df_cleaned.reset_index(drop=True)
        if args.readtype == 'batch':
            subset_df_cleaned = df_cleaned.iloc[600:1632]
        elif args.readtype == 'page':
            subset_df_cleaned = df_cleaned.iloc[0:775]
        else:
            subset_df_cleaned = df_cleaned
        combined_result = '\n'.join(subset_df_cleaned['Text'])

        # eval_prompt = '\n\n 위의 정보들 이용해서 진단 및 치료내역, 소견내용, 처리과정을 작성해줘.'
        # eval_stream = llm_inference(combined_result + eval_promplt, stream=False)

        final_prompt = '\n\n 위의 내용을 이용해서 아래 정보의 고객에 대한 자세한 손해사정보고서를 만들어줘.'
        final_stream = llm_inference(combined_result + final_prompt + user_info_prompt, stream=True)

        for chunk in final_stream:
          print(chunk['message']['content'], end='', flush=True)
        print('\n')

    print('Done')