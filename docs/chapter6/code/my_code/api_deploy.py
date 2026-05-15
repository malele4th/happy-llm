import os
import time
from huggingface_hub import get_token
from openai import OpenAI

api_key = get_token()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=api_key,
    timeout=120.0,
)


print(time.ctime(), "start")

completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B:featherless-ai",
    messages=[
        {
            "role": "user",
            "content": "中国的首都是哪个城市,介绍一下这个城市?"
        }
    ],
)

print(completion.choices[0].message)
print(time.ctime(), "end")

