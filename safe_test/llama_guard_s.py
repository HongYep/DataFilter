import json
from openai import OpenAI
import os
import time
import numpy as np
from tqdm import tqdm
LLAMA_GUARD_API_KEY = "token-pjlab"
LLAMA_GUARD_URL = "http://10.140.0.204:8614/v1"
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# import debugpy
# try:
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
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
                model = "/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-Guard-3-8B",
                messages=messages,
                max_tokens=10,
                logprobs=True,
                top_logprobs=5
            )
            for probs in response.choices[0].logprobs.content[1].top_logprobs:
                if probs.token == 'unsafe':
                    score = np.exp(probs.logprob)
            return score
        except Exception as e:
            print(f"llama_guard_eval Error:{e}")
            time.sleep(1)
            continue
    return 0

def data_process(data):
    if data['input'] == '':
        data['score'] = llama_guard_eval(data['instruction'], data['output'])
    else:
        data['score'] = llama_guard_eval(data['instruction']+'\n'+data['input'], data['output'])
    return data
if __name__ == '__main__':
    file_paths = [
        '/mnt/petrelfs/lihao1/DataFilter/data/alpaca_data_cleaned.json',
        '/mnt/petrelfs/lihao1/DataFilter/data/dolly-clean.json',
        
    ]
    for file_path in file_paths:
        data_list = []
        with open(file_path,'r' ) as f:
            data_list = json.load(f)
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [executor.submit(data_process, data) for data in data_list]
            

            processed_data = []
            for future in tqdm(as_completed(futures), total=len(data_list), desc=f"Processing {file_path}"):
                processed_data.append(future.result())

        processed_file_path = '/mnt/petrelfs/lihao1/DataFilter/data/'+ file_path.split('/')[-1].split('.')[0]+'_llama_score.json'
        with open(processed_file_path,'w') as f:
            json.dump(processed_data, f, indent=4)
            