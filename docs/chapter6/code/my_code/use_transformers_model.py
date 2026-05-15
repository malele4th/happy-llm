import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print(time.ctime(), "开始加载 tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
print(time.ctime(), "tokenizer 加载完成")

print(time.ctime(), "开始加载 model")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
print(time.ctime(), "model 加载完成")


messages = [
    {"role": "user", "content": "你是谁? 你有哪些功能"},
]
print(time.ctime(), "输入 messages:", messages)

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)
print(time.ctime(), "inputs 生成完成")

outputs = model.generate(**inputs, max_new_tokens=256)
print(time.ctime(), "推理结束, 输出:", tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

