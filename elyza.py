from vllm import LLM, SamplingParams

# llm = LLM(model="elyza/Llama-3-ELYZA-JP-8B-AWQ", quantization="awq")
repo_id="elyza/Llama-3-ELYZA-JP-8B-AWQ"
local_dir="/home/nocon/src/Llama-3-ELYZA-JP-8B-AWQ"
llm = LLM(model=local_dir, quantization="awq", max_model_len=4000, gpu_memory_utilization=0.9 )
# max_model_len=20000, gpu_memory_utilization=0.9 

tokenizer = llm.get_tokenizer()

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=1000)
messages_batch = [
    [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "古代ギリシャを学ぶ上で知っておくべきポイントは？"}
    ],
    [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": "クマが海辺に行ってアザラシと友達になり、最終的には家に帰るというプロットの短編小説を書いてください。"}
    ]
]

prompts = [
    tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    for messages in messages_batch
]

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    print(output.outputs[0].text)
    print("=" * 50)