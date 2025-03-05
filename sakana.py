import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# 1. load model
device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "SakanaAI/TinySwallow-1.5B-Instruct"

local_dir="/home/nocon/src/TinySwallow-1.5B-Instruct"
snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=["*.safetensor"])
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", torch_dtype=torch.bfloat16)

# model = AutoModelForCausalLM.from_pretrained(repo_id)
# tokenizer = AutoTokenizer.from_pretrained(repo_id)
model.to(device)

# 2. prepare inputs
# text = "知識蒸留について簡単に教えてください。"
text = "linuxでよく使われるaliasの設定例を教えてください。"
messages = [{"role": "user", "content": text}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

# 3. generate
output_ids = model.generate(
    input_ids.to(device),
    max_new_tokens=1024,
)
output_ids = output_ids[:, input_ids.shape[1] :]
generated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print(generated_text)
