import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# MODEL_NAME = "BAAI/bge-m3"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)
#
# local_model_path = "../model/embedding"
# tokenizer.save_pretrained(local_model_path)
# model.save_pretrained(local_model_path)

local_model_path = "../model/embedding"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE)


def encode(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return embeddings


documents = ["Hello, world!", "This is a test sentence.", "Another example sentence."]
query = "Test sentence"

# 문서 임베딩 생성
document_embeddings = encode(documents)

# FAISS 벡터 스토어 생성
dimension = document_embeddings.shape[1]  # 임베딩 벡터의 차원
index = faiss.IndexFlatL2(dimension)  # L2 거리 기반 인덱스 생성
index.add(document_embeddings)  # 임베딩 벡터 추가

# 쿼리 임베딩 생성 및 검색
query_embedding = encode([query])
D, I = index.search(query_embedding, k=3)  # k개의 가장 가까운 문서 검색

print("Indices of similar documents:", I)
print("Distances of similar documents:", D)
print("Similar documents:", [documents[i] for i in I[0]])