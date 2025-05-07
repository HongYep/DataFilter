import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
layer_num_start = 14
layer_num_end = 15
file_name = 'MetaMathQA_10000'
res_logits_str = f'llama3_{layer_num_start}2{layer_num_end}_{file_name}_avg100'
# res_logits_str = f'llama3_bi_res_24_{file_name}_avg100'
model_id = '/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct'
dir_id = 'llama3_sort_results'

model = AutoModelForCausalLM.from_pretrained(model_id,
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

with open(f'data/MetaMathQA_10000.json','r') as f:
    data = []
    target_data = json.load(f)
    
with open('data/pure-bad-100.jsonl', 'r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    unsafe_data = data

with open('data/pure-bad-100-anchor1.jsonl', 'r') as f:
    safe_data = []
    for line in f:
        safe_data.append(json.loads(line))

def alpaca_data_process(data, is_alpaca = True):
    if is_alpaca:
        if 'input' in data.keys() and data['input'] != '':
            message = [
                {"role": "user", "content": f"{data['instruction']}\n{data['input']}"},
                {"role": "assistant", "content": data['output']},
            ]
        else:
            message = [
                {"role": "user", "content": f"{data['instruction']}"},
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
        if layer_num_end - layer_num_start == 1:
            rep = model(input_ids).hidden_states[layer_num_start][0][-1].detach().cpu()
        else:
            rep = torch.stack(model(input_ids).hidden_states[layer_num_start:layer_num_end]).squeeze()[:,-1,:].flatten().detach().cpu()
    return rep

def safety_data_process(data):
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            data['messages'],
            return_tensors="pt"
        ).to(model.device)
        print(len(model(input_ids).hidden_states))
        print(model(input_ids).hidden_states[0].shape)
        if layer_num_end - layer_num_start == 1:
            rep = model(input_ids).hidden_states[layer_num_start][0][-1].detach().cpu()
        else:
            rep = torch.stack(model(input_ids).hidden_states[layer_num_start:layer_num_end]).squeeze()[:,-1,:].flatten().detach().cpu()
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
    

unsafe_rep = torch.stack([safety_data_process(data) for data in tqdm(unsafe_data, desc = 'unsafe_datas')])

safe_rep = torch.stack([safety_data_process(data) for data in tqdm(safe_data, desc = 'safe_datas')])

target_rep = torch.stack([alpaca_data_process(data) for data in tqdm(target_data , desc=f'target_datas:{file_name}')])


select_n = 1000
indices, scores = top_cosine_similarity(target_rep, unsafe_rep, safe_rep)

values = []
for i,s in zip(indices, scores):
    target_data[i].update({'sim_score': float(s)})
    values.append(target_data[i])

with open(f"{dir_id}/{res_logits_str}_mean.json",'w') as file:
    json.dump(values, file, indent=4)

sorted_by_score = sorted(values, key=lambda x: x['sim_score'], reverse=True)
remove_top_2000 = sorted_by_score[2000:] 

with open(f"{dir_id}/{res_logits_str}_remove_top_2000.json", 'w') as file:
    json.dump(remove_top_2000, file, indent=4)  

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
