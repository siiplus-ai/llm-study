import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

repo_id="meta-llama/Llama-3.2-3B-Instruct"
local_dir="/home/nocon/src/Llama-3.2-3B-Instruct"
snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=["*.safetensor"])

tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", torch_dtype=torch.bfloat16)

message = [
    {"role": "system", "content": "あなたは日本語ネイティブなアメリカの大統領のAIアシスタントです。"},
    {"role": "user", "content": "プログラミングが上達する方法を教えてください"},
 ]

prompt = tokenizer.apply_chat_template(message, tokenize=False)

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
answers = model.generate(
    **input_ids,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.1,
    eos_token_id=[
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ],
)
answer = tokenizer.decode(answers[0])
print(answer)
