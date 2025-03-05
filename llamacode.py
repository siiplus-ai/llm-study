# from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import transformers
import torch

repo_id="meta-llama/CodeLlama-7b-hf"
# model = "meta-llama/CodeLlama-7b-hf"

local_dir="/home/nocon/src/CodeLlama-7b-hf"
snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=["*.safetensor"])

tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", torch_dtype=torch.bfloat16)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to("cpu")

# tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer,
)

sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
