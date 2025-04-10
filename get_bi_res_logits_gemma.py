import torch
from transformers import Gemma3ForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='4b')
args = parser.parse_args()

res_logits_str = f"gemma_3_{args.model_size}_bi_res_logits_avg100"
model_id = f'/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-{args.model_size}-it'
dir_id = f'gemma_3_{args.model_size}_sort_results'

model = Gemma3ForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             output_hidden_states=True,
                                             return_dict_in_generate=True,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# import debugpy
# try:
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

with open('data/alpaca-cleaned/alpaca_data_cleaned.json','r') as f:
    data = []
    target_data = json.load(f)
    
with open('data/pure_bad_dataset/pure-bad-100.jsonl', 'r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    unsafe_data = data

with open('data/pure_bad_dataset/pure-bad-100-anchor1.jsonl', 'r') as f:
    safe_data = []
    for line in f:
        safe_data.append(json.loads(line))

def alpaca_data_process(data, is_alpaca = True):
    if is_alpaca:
        if data['input'] == '':
            message = [
                {"role": "user", "content": f"{data['instruction']}"},
                {"role": "assistant", "content": data['output']},
            ]
        else:
            message = [
                {"role": "user", "content": f"{data['instruction']}\n{data['input']}"},
                {"role": "assistant", "content": data['output']},
            ]
    else:                      
        if data['input'] == '':
            message = [
                {"role": "user", "content": f"{data['instruction']}"},
                {"role": "assistant", "content": data['output']},
            ]
        else:
            message = [
                {"role": "user", "content": f"{data['instruction']}\n{data['input']}"},
                {"role": "assistant", "content": data['output']},
            ]
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            message,
            return_tensors="pt"
        ).to(model.device)
        rep = model(input_ids).logits[0][-1].cpu()
    return rep

def unsafe_data_process(data):
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            data['messages'],
            return_tensors="pt"
        ).to(model.device)
        rep = model(input_ids).logits[0][-1].cpu()
    return rep

def top_cosine_similarity(A, B, C, avg_n = 100):
    A = A.to(torch.float32).numpy()
    B = B.to(torch.float32).numpy()
    C = C.to(torch.float32).numpy()
    unsafe_sim = cosine_similarity(A,B)
    safe_sim = cosine_similarity(A,C)
    final_sim = unsafe_sim - safe_sim
    sorted_similarity_matrix = np.sort(final_sim, axis=1)[:, ::-1]
    scores = np.mean(sorted_similarity_matrix[:, :avg_n], axis=1)
    rankings = np.argsort(scores)[::-1]
    return rankings, scores[rankings]
    

unsafe_rep = torch.stack([unsafe_data_process(data) for data in tqdm(unsafe_data, desc = 'unsafe_datas')])

safe_rep = torch.stack([unsafe_data_process(data) for data in tqdm(safe_data, desc = 'safe_datas')])

target_rep = torch.stack([alpaca_data_process(data) for data in tqdm(target_data , desc='target_datas')])


select_n = 1000
indices, scores = top_cosine_similarity(target_rep, unsafe_rep, safe_rep)

values = []
for i,s in zip(indices, scores):
    target_data[i].update({'sim_score': float(s)})
    values.append(target_data[i])

with open(f"{dir_id}/{res_logits_str}_mean.json",'w') as file:
    json.dump(values, file, indent=4)

top_indices = indices[:select_n]
top_scores = scores[:select_n]
top_values = []
for i,s in zip(top_indices, top_scores):
    target_data[i].update({'sim_score': float(s)})
    top_values.append(target_data[i])

with open(f"{dir_id}/{res_logits_str}_mean_top_{select_n}.json",'w') as file:
    json.dump(top_values, file, indent=4)

bottom_indices = indices[-select_n:]
bottom_scores = scores[-select_n:]
bottom_values = []

for i,s in zip(bottom_indices, bottom_scores):
    target_data[i].update({'sim_score': float(s)})
    bottom_values.append(target_data[i])

with open(f"{dir_id}/{res_logits_str}_mean_bottom_{select_n}.json",'w') as file:
    json.dump(bottom_values, file, indent=4)
