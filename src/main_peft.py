import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# 모델 및 토크나이저 설정
model_name = "ays-expanse-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

# LoRA 설정
lora_config = LoraConfig(
    r=16,  # 랭크 값
    lora_alpha=32,
    target_modules=["query", "value"],  # LoRA 적용 대상 모듈
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_int8_training(model)  # LoRA를 위해 모델 준비
model = get_peft_model(model, lora_config)  # LoRA로 모델 변환

# 데이터셋 다운로드 및 준비
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # 예: wikitext 데이터셋 사용

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 훈련 및 평가 데이터셋 분리
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

# 훈련 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 모델 훈련
trainer.train()

# 모델 평가
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# 텍스트 생성 테스트
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(inputs["input_ids"], max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Text: {generated_text}")
