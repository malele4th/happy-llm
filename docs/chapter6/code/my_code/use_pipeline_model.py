import time
from transformers import pipeline

print(time.ctime(), "开始加载 pipeline")
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
print(time.ctime(), "pipeline 加载完成")

messages = [
    {"role": "user", "content": "你是谁? 你有哪些功能"},
]

print(time.ctime(), "输入 messages:", messages)
output = pipe(messages)
print(time.ctime(), "推理结束, 输出:", output)

