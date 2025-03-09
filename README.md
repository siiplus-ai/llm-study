# hagging face

curl -v -H "Authorization: Bearer $hugging_token" -H "Connection: close" https://api-inference.huggingface.co/models

curl -v -H "Authorization: Bearer $hugging_token" -H "Connection: close" https://huggingface.co/meta-llama/CodeLlama-7b-hf


# lama3

## meta-llama/Llama-3.2-3B-Instruct

### source
git clone https://$hugging_user:$hugging_token@huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### lib
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers
pip install accelerate

py llama3.py

## meta-llama/Llama-3.2-1B

https://huggingface.co/meta-llama/Llama-3.2-1B

### source
git clone https://$hugging_user:$hugging_token@huggingface.co/meta-llama/Llama-3.2-1B



## meta-llama/CodeLlama-7b-hf

https://huggingface.co/meta-llama/CodeLlama-7b-hf

### source
git clone https://$hugging_user:$hugging_token@huggingface.co/meta-llama/CodeLlama-7b-hf


## SakanaAI/TinySwallow-1.5B

https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct

git clone https://huggingface.co/SakanaAI/TinySwallow-1.5B-Instruct


# ELYZA

https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-AWQ

pip install vllm

# llama.cpp

sudo apt install cmake

https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89"

## Use `CUDA_VISIBLE_DEVICES` to hide the first compute device.
CUDA_VISIBLE_DEVICES="-0" ./build/bin/llama-server --port 8080 --model /home/nocon/src/Llama-3-ELYZA-JP-8B-GGUF/Llama-3-ELYZA-JP-8B-q4_k_m.gguf

```
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    { "role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。" },
    { "role": "user", "content": "古代ギリシャを学ぶ上で知っておくべきポイントは？" }
  ],
  "temperature": 0.6,
  "max_tokens": -1,
  "stream": false
}'
```
# plamo-2

https://huggingface.co/pfnet/plamo-2-1b

numpy>=1.26.4
numba>=0.60.0
torch>=2.4.1
transformers>=4.44.2

mamba_ssm>=2.2.2
causal_conv1d>=1.4.0


# install causal_conv1d>=1.4.0

ModuleNotFoundError: No module named 'torch'



# ebpf

sudo apt update
sudo apt install -y bcc bpfcc-tools libbpf-dev llvm libelf-dev build-essential
pip install bcc
