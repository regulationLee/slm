import torch
import transformers
import accelerate as Accelerator
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DefaultDataCollator
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig

from datasets import load_dataset

from huggingface_hub import login

login(token="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")

model_name = "CohereForAI/aya-23-8b"

QUANTIZE_4BIT = True  # 모델을 4비트로 양자화(Quantize)할지 여부를 설정합니다. 양자화는 모델의 메모리 사용량을 줄이고, 일부 경우에 성능을 향상시킬 수 있습니다.
USE_GRAD_CHECKPOINTING = True  # 그래디언트 체크포인팅을 사용할지 여부를 설정합니다. 이는 메모리 사용량을 줄여주지만, 훈련 속도는 느려질 수 있습니다.
TRAIN_BATCH_SIZE = 16  # 훈련 시 사용되는 배치 크기를 설정합니다. 이 값은 사용 가능한 GPU 메모리에 따라 조정될 수 있습니다.
TRAIN_MAX_SEQ_LENGTH = 256  # 훈련 시 사용되는 최대 시퀀스 길이를 설정합니다. 시퀀스 길이가 길어질수록 메모리 사용량이 증가합니다.
USE_FLASH_ATTENTION = True
GRAD_ACC_STEPS = 2  # 그래디언트 누적 단계 수를 설정합니다. 이는 작은 배치 크기로 훈련할 때 유용하며, 효과적으로 배치 크기를 늘려줍니다.

quantization_config = None
if QUANTIZE_4BIT:
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,  # 모델을 4비트로 양자화하여 로드합니다.
      bnb_4bit_quant_type="nf4",  # 4비트 양자화 유형을 'nf4'로 설정합니다.
      bnb_4bit_use_double_quant=True,  # 이중 양자화를 사용합니다.
      bnb_4bit_compute_dtype=torch.bfloat16,  # 계산 시 사용할 데이터 타입을 bfloat16으로 설정합니다.
  )

attn_implementation = None

model = AutoModelForCausalLM.from_pretrained(
    model_name,  # 사전 학습된 모델의 이름을 지정합니다.
    quantization_config=quantization_config,  # 양자화 설정을 적용합니다 (양자화가 활성화된 경우).
    attn_implementation=attn_implementation,  # 주의 메커니즘(Attention Mechanism) 구현 설정을 적용합니다.
    torch_dtype=torch.bfloat16,  # 모델의 데이터 타입을 bfloat16으로 설정합니다.  # 사용 가능한 모든 GPU를 자동으로 매핑하여 분산 훈련을 설정합니다.
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

from datasets import load_dataset

# SQuAD 데이터셋 로드
squad_dataset = load_dataset("squad")

# 변환 함수 정의
def squad_to_translated_wikiqa(example):
    # SQuAD의 context와 question을 사용하여 inputs 생성
    inputs = f"{example['question']}."

    # SQuAD의 answers에서 첫 번째 답변을 사용하여 targets 생성
    targets = f"{example['answers']} \n {example['context']}"

    # translated_wikiqa 형식에 맞게 데이터 구성
    return {
        "id": example["id"],  # SQuAD의 id를 그대로 사용
        "inputs": inputs,
        "targets": targets,
        "dataset_name": "SQuAD",
        "sub_dataset_name": "-",
        "task_type": "question-answering",
        "template_id": 1,
        "language": "eng",  # "dan"을 고정값으로 설정
        "script": "Latn",  # 라틴 문자로 고정
        "split": "train"  # 현재는 train 데이터셋만 처리
    }

# SQuAD 데이터셋을 translated_wikiqa 형태로 변환
translated_wikiqa_train = squad_dataset["train"].map(squad_to_translated_wikiqa)
translated_wikiqa_validation = squad_dataset["validation"].map(squad_to_translated_wikiqa)

"""
data = load_dataset("squad", split="train[:5000]")

data = data.train_test_split(test_size=0.2)
data["train"][0]

dataset = load_dataset("CohereForAI/aya_collection", "translated_wikiqa")['train']
dataset = dataset.filter(lambda example: example['language']=='dan')
"""


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['inputs'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{example['inputs'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['targets'][i]}"
        output_texts.append(text)
    return output_texts

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    gradient_checkpointing=USE_GRAD_CHECKPOINTING,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate= 3e-5,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",
)

peft_config = LoraConfig(
    lora_alpha=16,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=translated_wikiqa_train,
    peft_config=peft_config,
    max_seq_length=TRAIN_MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=formatting_prompts_func
)

# trainer.train()
# trainer.model.save_pretrained(save_directory='./results')
# model.config.use_cache = True

model.eval()

def get_message_format(prompts):
  messages = []

  for p in prompts:
      messages.append(
        [{"role": "user", "content": p}]
      )
  return messages

def generate_aya(
      model,
      prompts,
      temperature=0.75,
      top_p=1.0,
      top_k=0,
      max_new_tokens=1024
    ):

  messages = get_message_format(prompts)

  input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
      )

  input_ids = input_ids.to(model.device)
  prompt_padded_len = len(input_ids[0])

  gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
      )

  gen_tokens = [
      gt[prompt_padded_len:] for gt in gen_tokens
  ]

  gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
  return gen_text

quantization_config = None
if QUANTIZE_4BIT:
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
  )

attn_implementation = None

loaded_sft_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

loaded_sft_model.load_adapter('./results/checkpoint-8200')

prompts = [
    'Tell me about the University of Notre Dame'
]

generations = generate_aya(loaded_sft_model, prompts)

for p, g in zip(prompts, generations):
  print(
      "PROMPT", p, "RESPONSE", g, "\n", sep="\n"
  )