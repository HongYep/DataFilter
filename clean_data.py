import json
from openai import OpenAI
import os
import time
import numpy as np
from tqdm import tqdm
LLAMA_GUARD_API_KEY = "token-pjlab"
LLAMA_GUARD_URL = "http://10.140.1.30:8612/v1"

def llama_guard_eval(query, res = None):
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": res}
    ] if res is not None else [
        {"role": "user", "content": query}
    ]
    for _ in range(3):
        try:
            if 'http_proxy' in os.environ:
                del os.environ['http_proxy'], os.environ['HTTP_PROXY']
            llama_guard_client = OpenAI(base_url=LLAMA_GUARD_URL, api_key=LLAMA_GUARD_API_KEY)
            response = llama_guard_client.chat.completions.create(
                model = "/mnt/hwfile/trustai/models/Meta-Llama-Guard-2-8B",
                messages=messages,
                max_tokens=10
            )
            return response.choices[0].message.content.split('\n')[0]
        except Exception as e:
            print(f"llama_guard_eval Error:{e}")
            time.sleep(1)
            continue
    return 0

if __name__ == '__main__':
    count = 0
    data_list = []
    with open('data/alpaca-gpt4.json', 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    final_list = []
    for data in tqdm(data_list):
        res = llama_guard_eval(data['input'], data['output'])
        if res == 'unsafe':
            count+=1
            print(count)
            continue
        final_list.append(data)
    with open('data/alpaca-gpt4-llama-clean.json','r') as f:
        json.dump(final_list, f, indent=4)
        