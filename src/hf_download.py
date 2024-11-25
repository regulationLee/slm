import os
from huggingface_hub import snapshot_download

model_id = "bitext/Mistral-7B-Insurance"
local_model_path = os.path.join(os.getcwd(), "models", model_id.replace("/", "_"))
if not os.path.exists(local_model_path):
    os.mkdir(local_model_path)
snapshot_download(repo_id=model_id, local_dir=local_model_path, local_dir_use_symlinks=False, revision="main")