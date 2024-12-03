import os
import torch
import json  # JSON 형식의 파일을 읽고 쓰기 위해 사용되는 표준 라이브러리
import random  # 무작위 작업을 수행하기 위한 표준 라이브러리
from tqdm import tqdm  # 반복문 진행 상황을 시각적으로 표시해 주는 라이브러리

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
import accelerate as acc
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb


def get_message_format(prompts):
    messages = []  # 메시지를 저장할 빈 리스트를 초기화합니다.

    for p in prompts:  # 주어진 모든 프롬프트에 대해 반복합니다.
        messages.append(
        [{"role": "user", "content": p}]  # 각 프롬프트를 딕셔너리 형식으로 변환하여 리스트에 추가합니다.
      )

    return messages  # 변환된 메시지 리스트를 반환합니다.

def generate_aya_23(
      prompts,
      model,
      temperature=0.3,
      top_p=0.75,
      top_k=0,
      max_new_tokens=1024
    ):
    """
    주어진 프롬프트를 사용하여 모델을 통해 텍스트를 생성하는 함수입니다.

    인자:
    - prompts: 텍스트 프롬프트의 리스트
    - model: 사전 학습된 언어 모델
    - temperature: 생성의 다양성을 제어하는 파라미터 (0.3 기본값)
    - top_p: 상위 p% 확률의 토큰만 고려하는 파라미터 (0.75 기본값)
    - top_k: 상위 k개의 토큰만 고려하는 파라미터 (0 기본값)
    - max_new_tokens: 생성할 최대 토큰 수 (1024 기본값)
    """

    # 프롬프트를 메시지 형식으로 변환합니다.
    messages = get_message_format(prompts)

    # 메시지를 토크나이저를 사용하여 입력 ID로 변환합니다.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
      )

    # 입력 ID를 모델의 장치로 이동시킵니다.
    input_ids = input_ids.to(model.device)
    prompt_padded_len = len(input_ids[0])  # 패딩된 입력 ID의 길이를 저장합니다.

    # 모델을 사용하여 텍스트를 생성합니다.
    gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
      )

    # 생성된 토큰 중에서 입력 프롬프트 이후의 토큰만 가져옵니다.
    gen_tokens = [
      gt[prompt_padded_len:] for gt in gen_tokens
    ]

    # 생성된 토큰을 텍스트로 디코딩합니다.
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    return gen_text  # 생성된 텍스트를 반환합니다.

model_name = "CohereForAI/aya-expanse-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

QUANTIZE_4BIT = True
USE_GRAD_CHECKPOINTING = True
TRAIN_BATCH_SIZE = 8
TRAIN_MAX_SEQ_LENGTH = 512
GRAD_ACC_STEPS = 4
attn_implementation = None

quantization_config = None
if QUANTIZE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 모델 적용
model = get_peft_model(model, lora_config)

# 데이터셋 다운로드 및 준비
# JSON 파일 경로
json_path = 'EN_TO_KO.json'

# JSON 파일을 열고 텍스트 항목을 읽습니다.
with open(json_path, encoding='utf-8') as f:
    items = [json.loads(line)['text'] for line in tqdm(f)]  # 각 라인을 JSON으로 로드하고, 'text' 필드를 추출하여 리스트에 저장

# 영어와 한국어 문장을 분리하는 함수 정의
def split_eng_kor(text):
    # 텍스트를 줄바꿈 문자로 분리하여 소스와 타겟을 구분
    lines = text.split('\n')

    # 적절한 줄에서 소스와 타겟을 추출
    eng = lines[0].replace('### source: ', '').strip()  # 소스 텍스트에서 '### source: ' 제거하고 공백 제거
    kor = lines[2].replace('### target: ', '').strip()  # 타겟 텍스트에서 '### target: ' 제거하고 공백 제거
    return eng, kor  # 영어와 한국어 텍스트 반환

# 각 항목을 영어와 한국어 문장으로 분리
items = [split_eng_kor(e) for e in tqdm(items)]  # 각 항목에 대해 split_eng_kor 함수 적용

# 항목들을 무작위로 섞기
random.shuffle(items)

# 처음 20,000개의 항목을 입력과 타겟 리스트로 분리
inputs = [e for e, k in tqdm(items[:1000])]  # 처음 20,000개의 입력 텍스트 추출
targets = [k for e, k in tqdm(items[:1000])]  # 처음 20,000개의 타겟 텍스트 추출

# 입력과 타겟을 포함하는 데이터셋 생성
dataset = Dataset.from_dict({'inputs': inputs, 'targets': targets})

# 영어와 한국어 문장을 포맷팅하는 함수 정의
def convert_to_input_text(eng, kor):
    return f'Translate from English to Korean: {eng}{kor}'  # 주어진 형식으로 포맷팅

# 데이터셋의 입력과 타겟을 포맷팅하는 함수 정의
def formatting_prompts_func(dataset):
    inputs = dataset['inputs']  # 데이터셋에서 입력 리스트 추출
    targets = dataset['targets']  # 데이터셋에서 타겟 리스트 추출
    return [convert_to_input_text(e, k) for e, k in tqdm(zip(inputs, targets), total=len(inputs))]  # 각 입력과 타겟 쌍에 대해 포맷팅 함수 적용

# 훈련 설정
training_arguments = TrainingArguments(
    output_dir="results",  # 훈련 결과를 저장할 디렉토리
    num_train_epochs=10,  # 훈련 에포크 수
    per_device_train_batch_size=TRAIN_BATCH_SIZE,  # 각 장치에서의 훈련 배치 크기
    optim="paged_adamw_32bit",  # 옵티마이저 설정
    save_steps=200,  # 체크포인트 저장 주기(스텝 수)
    logging_steps=10,  # 로깅 주기(스텝 수)
    learning_rate=1e-3,  # 학습률
    weight_decay=0.001,  # 가중치 감쇠
    fp16=False,  # 16비트 부동 소수점 사용 여부
    bf16=True,  # bfloat16 사용 여부
    warmup_ratio=0.05,  # 워밍업 비율
    group_by_length=True,  # 시퀀스 길이에 따라 그룹화 여부
    lr_scheduler_type="constant",  # 학습률 스케줄러 유형
    report_to="none"  # 로깅을 보고할 대상
)

trainer = SFTTrainer(
    model=model,  # 훈련할 모델
    train_dataset=dataset,  # 훈련 데이터셋
    peft_config=lora_config,  # PEFT 설정
    max_seq_length=TRAIN_MAX_SEQ_LENGTH,  # 최대 시퀀스 길이
    tokenizer=tokenizer,  # 토크나이저
    args=training_arguments,  # 훈련 인자
    formatting_func=formatting_prompts_func  # 프롬프트 포맷팅 함수
)

# 모델 훈련
trainer.train()

trainer.model.save_pretrained(save_directory='aya-qlora')
model.config.use_cache = True
model.eval()

# 모델과 LoRA 어댑터 로드
quantization_config = None  # 양자화 설정을 위한 변수를 초기화합니다.
if QUANTIZE_4BIT:  # QUANTIZE_4BIT 플래그가 True일 경우 양자화 설정을 구성합니다.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 모델을 4비트로 양자화하여 로드합니다.
        bnb_4bit_quant_type="nf4",  # 4비트 양자화 유형을 'nf4'로 설정합니다.
        bnb_4bit_use_double_quant=True,  # 이중 양자화를 사용합니다.
        bnb_4bit_compute_dtype=torch.bfloat16,  # 계산 시 사용할 데이터 타입을 bfloat16으로 설정합니다.
    )

# 사전 학습된 언어 모델을 로드합니다.
loaded_model = AutoModelForCausalLM.from_pretrained(
    model_name,  # 사전 학습된 모델의 이름을 지정합니다.
    quantization_config=quantization_config,  # 양자화 설정을 적용합니다 (양자화가 활성화된 경우).
    attn_implementation=attn_implementation,  # 주의 메커니즘(Attention Mechanism) 구현 설정을 적용합니다.
    torch_dtype=torch.bfloat16,  # 모델의 데이터 타입을 bfloat16으로 설정합니다.
    device_map="auto",  # 사용 가능한 모든 GPU를 자동으로 매핑하여 분산 훈련을 설정합니다.
)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 어댑터 로드
loaded_model.load_adapter("aya-qlora")

# 테스트용 프롬프트 생성
prompts = [
    'Translate from English to Korean: Rates are competitive, almost always the best in the market',  # 첫 번째 프롬프트 (영어 -> 한국어 번역)
    'Translate from English to Korean: Two far-right Israeli ministers threaten to topple the government if it accepts Biden peace plan'  # 두 번째 프롬프트 (영어 -> 한국어 번역)
]

print('Before training')

generations = generate_aya_23(prompts, model)

# 각 프롬프트와 생성된 응답을 쌍으로 출력
for p, g in zip(prompts, generations):  # 각 프롬프트와 생성된 텍스트 쌍에 대해 반복
    print(
        "PROMPT", p, "RESPONSE", g, "\n", sep="\n"  # 프롬프트와 응답을 출력. 각 요소는 새로운 줄에 출력됩니다.
    )

print('\nAfter training')

# 프롬프트를 사용하여 텍스트 생성
generations = generate_aya_23(prompts, loaded_model)

# 각 프롬프트와 생성된 응답을 쌍으로 출력
for p, g in zip(prompts, generations):  # 각 프롬프트와 생성된 텍스트 쌍에 대해 반복
    print(
        "PROMPT", p, "RESPONSE", g, "\n", sep="\n"  # 프롬프트와 응답을 출력. 각 요소는 새로운 줄에 출력됩니다.
    )
