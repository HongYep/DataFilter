import json
from openai import OpenAI
import os
import time
import numpy as np
from tqdm import tqdm
LLAMA_GUARD_API_KEY = "token-pjlab"
LLAMA_GUARD_URL = "http://10.140.54.12:8612/v1"
from concurrent.futures import ThreadPoolExecutor
import threading
def llama_guard_eval(query, res):
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": res}
    ]
    for _ in range(3):
        try:
            if 'http_proxy' in os.environ:
                del os.environ['http_proxy'], os.environ['HTTP_PROXY']
            llama_guard_client = OpenAI(base_url=LLAMA_GUARD_URL, api_key=LLAMA_GUARD_API_KEY)
            response = llama_guard_client.chat.completions.create(
                model = "/mnt/petrelfs/share_data/safety_verifier/Llama-Guard-3-8B",
                messages=messages,
                max_tokens=10,
                # temperature=0.0,
            )
            return response.choices[0].message.content.split('\n')[2]
        except Exception as e:
            print(f"llama_guard_eval Error:{e}")
            time.sleep(1)
            continue
    return 0
results_lock = threading.Lock()
processed_outputs = []

def evaluate_output(index_output_pair):
    index, output = index_output_pair
    score = llama_guard_eval(output['instruction'], output['output'])
    with results_lock:
        output['score'] = score
        return (index, output)

for root, dirs, files in os.walk('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/med_results'):
    print("WALK",root, dirs, files)
    # with open('/mnt/petrelfs/lihao1/vlmmoe/safe_test/MM-processed_questions/total.json','r') as f:
    #     total = json.load(f)
    for file in files:
        if 'matamath' not in file:
            continue
        file = os.path.join(root, file)
        with open(file, 'r') as f:
            output_list = json.load(f)
        if 'score' in output_list[0].keys():
            continue
        with ThreadPoolExecutor(max_workers=20) as executor:
            # 准备输入数据，保持索引信息
            indexed_outputs = list(enumerate(output_list))
            # 使用executor.map保持结果的顺序
            for result in tqdm(executor.map(evaluate_output, indexed_outputs),
                            total=len(output_list),
                            desc=file):
                index, output = result
                output_list[index] = output
        with open(file, 'w') as f:
            json.dump(output_list, f, indent=4)