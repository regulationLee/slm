#### original code from https://www.ncloud-forums.com/topic/318/
from __future__ import print_function
import os
import io
from tqdm import tqdm
import numpy as np
from collections import defaultdict
# import easyocr
import re
# from pypdf import PdfReader
import fitz
# import pytesseract
from pdf2image import convert_from_path
import glob
from PIL import Image, ImageEnhance, ImageFilter
import tifffile
import numpy
from llama_index.core import SimpleDirectoryReader

from llama_index.readers.file import (
    DocxReader,
    HWPReader,
)
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
from langchain_core.documents import Document
import pymupdf4llm
from pathlib import Path


def PIL2array(img):
    """ Convert a PIL/Pillow image to a numpy array """
    return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0], 3)


def parse_input(input):
    personal_info = defaultdict(list)
    personal_info['name'] = '고객이름 ' + input.name
    personal_info['age'] = '나이 ' + input.age
    personal_info['start'] = '의뢰일 ' + input.start
    personal_info['end'] = '완료일 ' + input.end
    personal_info['poc'] = '심사자 ' + input.poc
    personal_info['inc'] = '확인회사 ' + input.inc
    personal_info['supervisor'] = '확인자 ' + input.supervisor
    personal_info['leader'] = '팀장 ' + input.leader
    personal_info['contract'] = '계약사항 ' + input.contract
    personal_info['connum'] = '계약번호 ' + input.connum
    personal_info['conday'] = '계약일자 ' + input.conday
    personal_info['constatus'] = '계약상태 ' + input.constatus

    return personal_info


def read_doc_report(path):
    ################# Read DOCX file and print #################
    loader = DocxReader()
    documents = loader.load_data(path)
    return documents


def read_pdf_files(path):
    ############### PDF file reader #################
    loader = PyMuPDFLoader(file_path=path)
    documents = loader.load()
    full_text = pymupdf4llm.to_markdown(path)
    full_text = full_text.replace('\n', ' ').strip()

    def remove_newlines_except_after_period(text):
        return re.sub(r'(?<!\.)(\n|\r\n)', ' ', text)

    def process_pages(pages: List[Document]) -> List[Document]:
        return [Document(page_content=remove_newlines_except_after_period(page.page_content), metadata=page.metadata)
                for page in pages]

    processed_data = process_pages(documents)

    # full_text = ""
    # for doc in documents:
    #     content = doc.page_content
    #     if content.strip():
    #         full_text += content

    return processed_data


def read_records(path):
    pdf_document = fitz.open(path)
    num_pages = pdf_document.page_count
    extracted_text = []

    for page_num in tqdm(range(num_pages)):
        page = pdf_document[page_num]

        # upscale
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes()

        # Convert to Grey scale
        image = Image.open(io.BytesIO(img_data)).convert('L')

        # convert to binary
        binary_image = image.point(lambda x: 0 if x < 128 else 255, '1')

        # Noise filter
        processed_image = binary_image.filter(ImageFilter.MedianFilter())

        # # Convert to RGB
        # rgb_image = processed_image.convert('RGB')
        #
        # enhancer = ImageEnhance.Contrast(rgb_image)
        # enhanced_image = enhancer.enhance(2)

        # OCR config
        config = ('-l kor+eng --oem 3 --psm 6')

        text = pytesseract.image_to_string(processed_image, config=config)
        extracted_text.append(text)

    full_text = "\n".join(extracted_text)

    # # save full text as file
    # with open("extracted_text.txt", "w") as text_file:
    #     text_file.write(full_text)

    return full_text


def read_records_by_page(page, easyocr_cond):
    if easyocr_cond:
        reader = easyocr.Reader(['ko', 'en'], gpu=True)

    # upscale
    pix = page.get_pixmap(dpi=300)
    img_data = pix.tobytes()

    # Convert to Grey scale
    image = Image.open(io.BytesIO(img_data)).convert('L')

    # convert to binary
    binary_image = image.point(lambda x: 0 if x < 128 else 255, '1')

    # Noise filter
    processed_image = binary_image.filter(ImageFilter.MedianFilter())

    if easyocr_cond:
        rgb_image = processed_image.convert('RGB')
        enhancer = ImageEnhance.Contrast(rgb_image)
        enhanced_image = enhancer.enhance(2)
        img_np = np.array(enhanced_image)
        text = reader.readtext(img_np, detail=0, text_threshold=0.7, low_text=0.4)
    else:
        # OCR config
        config = ('-l kor+eng --oem 3 --psm 6')
        text = pytesseract.image_to_string(processed_image, config=config)

    return text


    # pdf_file = open(file_path, 'rb')
    # pdf_reader = PdfReader(pdf_file)
    #
    # # 페이지 수
    # page_num = len(pdf_reader.pages)
    #
    # # text 추출
    # text = ''
    # for pn in range(page_num):
    #     page = pdf_reader.pages[pn]
    #     text += page.extract_text()
    #     print('debug')
    #
    # print(text)
def pdf_to_tiff(conf, input_file_name):
    ################# Read PDF file and convert to tiff #################
    file_name = input_file_name[:-4]
    file_path = os.path.join(conf.data_path, input_file_name)
    pages = convert_from_path(file_path)

    resize_ratio = conf.tiff_ratio

    for i, page in enumerate(pages):
        # resize images
        new_size = (int(page.width * resize_ratio), int(page.height * resize_ratio))
        resized_image = page.resize(new_size, Image.Resampling.LANCZOS)

        output_file = os.path.join(conf.result_path, f"{file_name}_{i}.tiff")
        resized_image.save(output_file, format="TIFF", compression="jpeg", quality=30)  # 품질 30%

    ###### Convert single tiff to multi tiff
    def merge_tiffs(input_files, output_file):
        images = [Image.open(file) for file in input_files]

        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            compression="tiff_deflate"  # 압축 옵션 설정 (선택 사항)
        )

    filelist = glob.glob(conf.result_path + "*.tiff")
    filelist.sort()
    output_file = conf.result_path + f"{file_name}_multitiff.tiff"

    merge_tiffs(filelist, output_file)


# ################# hwp file reader
# hwp_path = Path("/Users/user/Desktop/connector/hwp/report1.hwp")
# reader = HWPReader()
# documents = reader.load_data(file=hwp_path)
#
# print(documents)