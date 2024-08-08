#### original code from https://www.ncloud-forums.com/topic/318/

import os
from pypdf import PdfReader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import (
    DocxReader,
    HWPReader,
    PyMuPDFReader,
)
from pathlib import Path

DATA_PATH = '/home/glee/sakak/data/TSA/samples'
RESULT_PATH = '/home/glee/sakak/slm/results'

################# Docx file reader
file_name = '0-1. 손해사정 보고서_ABC_학습용.docx'
file_path = os.path.join(DATA_PATH, file_name)
loader = DocxReader()
documents = loader.load_data(file_path)
print(documents)


################# PDF file reader
# file_name = 'regulationLee.pdf'
# file_path = os.path.join(DATA_PATH, file_name)
# # loader = PyMuPDFReader()
# # documents = loader.load_data(file_path=file_path, metadata=True)
# # print([doc for doc in documents if doc.metadata.get('source') == '8']) # 8쪽만 확인


# pdf_file = open(file_path, 'rb')
# pdf_reader = PdfReader(pdf_file)

# # 페이지 수
# page_num = len(pdf_reader.pages)

# # text 추출
# text = ''
# for pn in range(3):
#     page = pdf_reader.pages[pn]
#     text += page.extract_text()

# print(text)


# ################# hwp file reader
# hwp_path = Path("/Users/user/Desktop/connector/hwp/report1.hwp")
# reader = HWPReader()
# documents = reader.load_data(file=hwp_path)
#
# print(documents)