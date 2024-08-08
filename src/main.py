from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

# 모델 이름 정의
model_path = "/home/glee/sakak/slm/models/Meta-Llama-3.1-70B-Instruct"
#model_path = "/home/glee/sakak/slm/models/Meta-Llama-3.1-70B"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# Accelerator 설정
accelerator = Accelerator(split_batches=True, dispatch_batches=True)

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_path, token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,  # fp16 precision으로 로드
    device_map="auto",  # 자동으로 GPU에 분배
    token=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 모델과 토크나이저를 accelerator로 준비
model, tokenizer = accelerator.prepare(model, tokenizer)

# 입력 텍스트 정의
input_texts = "상대성이론에 대해 설명해줘"

# 입력 텍스트를 토큰화
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=500)
#inputs = tokenizer(input_texts, return_tensors="pt").to('cuda')

# accelerator로 입력을 준비
inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

# 모델 예측
with torch.no_grad():
    outputs = accelerator.unwrap_model(model).generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        repetition_penalty = 1.2,
        top_k = 50,
        top_p = 0.9,
        temperature=0.7
    )

# 결과 디코딩
generated_texts = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 결과 출력
print(f"Generated Text: {generated_texts}")


# 메모리 해제
torch.cuda.empty_cache()


'''



import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

'''

