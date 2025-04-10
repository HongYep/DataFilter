import json
from openai import OpenAI
import os
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# import debugpy
# try:
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
os.environ["http_proxy"] = "http://10.1.20.57:23128"
os.environ["https_proxy"] = "http://10.1.20.57:23128"


def moderation_eval(query, res):
    for _ in range(3):
        try:
            client = OpenAI(api_key="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")
            response = client.moderations.create(
                model="text-moderation-latest",
                input=f"user query:{query} \n response:{res}",
            )
            # print(response)
            return response.results[0].category_scores.model_extra
        except Exception as e:
            print(f"moderation_eval Error:{e}")
            time.sleep(1)
            continue
    return 0

def data_process(data):
    if data['input'] == '':
        data['score'] = moderation_eval(data['instruction'], data['output'])
    else:
        data['score'] = moderation_eval(data['instruction']+'\n'+data['input'], data['output'])
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
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有任务
            futures = [executor.submit(data_process, data) for data in data_list]
            # 使用tqdm显示进度
            processed_data = []
            for future in tqdm(as_completed(futures), total=len(data_list), desc=f"Processing {file_path}"):
                processed_data.append(future.result())
        # 按原始顺序保存结果（如果需要保持顺序）
        # 注意：as_completed是按完成顺序返回的，所以这里需要重新排序
        # 如果顺序不重要，可以跳过这步
        # processed_data.sort(key=lambda x: data_list.index(x))
        processed_file_path = '/mnt/petrelfs/lihao1/DataFilter/data/'+ file_path.split('/')[-1].split('.')[0]+'_moderation_score.json'
        with open(processed_file_path,'w') as f:
            json.dump(processed_data, f, indent=4)