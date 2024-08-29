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
import itertools


from text_parser import *
from poc_medical_record import *

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain_core.documents import Document
import pymupdf4llm
import pdfplumber


if platform.system() == 'Linux':
    DATA_PATH = '/home/glee/sakak/data/TSA/samples/'
    RESULT_PATH = '/home/glee/sakak/slm/results/'
    DEVICE = 'cuda'
else:
    DATA_PATH = '/Users/glee/Desktop/sakak/TSA/samples/'
    RESULT_PATH = '/Users/glee/Desktop/sakak/slm/results/'
    DEVICE = 'mps'


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


def read_pdf_files(path):
    def extract_tables_from_pdf(pdf_path):
        tables = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 페이지에서 표 추출
                table = page.extract_table()
                if table:
                    # 추출된 표를 Pandas DataFrame으로 변환
                    df = pd.DataFrame(table[1:], columns=table[0])
                    tables.append(df)
        return tables

    ############### PDF file reader #################
    loader = PyMuPDFLoader(file_path=path)
    documents = loader.load()
    tables = extract_tables_from_pdf(path)
    filtered_tables = tables[7:21]
    text_dict = defaultdict()
    text = ""
    for i, table in enumerate(filtered_tables):
        if isinstance(table, pd.DataFrame):
            # table = table.fillna(method='ffill')
            table = table.ffill()
            table = table.apply(lambda x: table.columns + ":" + x.astype(str), axis=1)
            tmp_text = table.to_csv(index=False, header=False)
            # tmp_text = tmp_text.replace('\n', ' ').strip()
            tmp_text = re.sub(r'\n"보 장', 'TEMP_REPLACE', tmp_text)
            tmp_text = re.sub(r'\n보 장', 'TEMP_2REPLACE', tmp_text)
            tmp_text = re.sub(r'\n', ' ', tmp_text)
            tmp_text = re.sub(r'TEMP_REPLACE', '\n"보 장', tmp_text)
            tmp_text = re.sub(r'TEMP_2REPLACE', '\n"보 장', tmp_text)
            text_dict[i] = tmp_text.split("\n")
            text += tmp_text

    def process_pages(pages):
        return [Document(page_content=page, metadata=dict(page=i)) for i, page in enumerate(pages)]

    # processed_data = process_pages(documents)
    text_list = list(itertools.chain(*text_dict.values()))
    processed_data = process_pages(text_list)

    # full_text = ""
    # for doc in documents:
    #     content = doc.page_content
    #     if content.strip():
    #         full_text += content

    return processed_data


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

    print("=" * 50)
    # final_prompt = '위의 손해사정보고서의 스타일로 아래의 조사기록을 분석해서 보험금 지급 여부 판단을 포함한 손해사정보고서 작성'
    final_prompt = '위의 손해사정보고서의 스타일로 아래의 조사기록을 분석해서 손해사정보고서 작성'
    final_stream = llm_inference(final_prompt + user_info_prompt + diagnosis_str, stream=True)

    contents_output = ''
    for chunk in final_stream:
        content = chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
        contents_output += content
    print('\n')

    print("=" * 50)
    print('보험 약관 데이터 분석 중')

    record_file = '상해_상품안내서.pdf'
    file_path = os.path.join(DATA_PATH, record_file)
    contents = read_pdf_files(file_path)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # splits = text_splitter.split_documents(contents)
    # print(f'len(splits[0].page_content) : {len(splits[0].page_content)}')

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True},
    )
    vector_store = FAISS.from_documents(contents, embeddings)
    relevant_docs = vector_store.similarity_search(f'다음의 의료기록에 해당하는 내용을 찾아줘 \n\n {contents_output}', k=3)
    insurance_context = " ".join(doc.page_content for doc in relevant_docs)

    doc_prompt = f'Context: {insurance_context} \n Question: 이 보험 약관 데이터를 바탕으로 다음의 손해사정보고서의 질병 또는 부상정보별로 보험금 지급 여부 및 예상금액 판단 \n {contents_output}'
    input_prompt = doc_prompt

    # result_stream = llm_inference(insurance_context + input_prompt + diagnosis_str, stream=True)
    result_stream = llm_inference(input_prompt, stream=True)

    for chunk in result_stream:
        content = chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)
    print('\n')

    print("=" * 50)
    final_prompt = f'{contents_output} \n\n 위의 손해사정보고서 내용을 일자별로 개조식으로 요약.'
    final_stream = llm_inference(final_prompt, stream=True)

    for chunk in final_stream:
        print(chunk['message']['content'], end='', flush=True)
    print('\n')
