import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

repo_id="pfnet/plamo-2-1b"
local_dir="/home/nocon/src/plamo-2-1b"
snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=["*.safetensor"])

tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True)
print(pipeline("The future of artificial intelligence technology is ", max_new_tokens=32))

#from transformers import AutoTokenizer, AutoModelForCausalLM
#tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-2-1b", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("pfnet/plamo-2-1b", trust_remote_code=True)

text = "これからの人工知能技術は"
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_tokens = model.generate(
    inputs=input_ids,
    max_new_tokens=32,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
)[0]

generated_text = tokenizer.decode(generated_tokens)
print(generated_text)