import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

res_logits = True
model_id = '/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct'
dir_id = 'llama_sort_results'
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.float16,
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

def safe_data_process(data, is_alpaca = True):
    if is_alpaca:
        if data['input'] == '':
            message = [
                {"role": "user", "content": f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{data['instruction']}\n\n### Response:\n"},
            ]
        else:
            message = [ 
                {"role": "user", "content": f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:\n"},
            ]
    else:
        message = [
            {"role": "system", "content": data['instruction']},
            {"role": "user", "content": data['input']},
        ]
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=res_logits,
            return_tensors="pt"
        ).to(model.device)
        logits = model(input_ids).hidden_states[-1][0][-1].cpu()
    return logits

def unsafe_data_process(data):
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            data['messages'][:1],
            add_generation_prompt=res_logits,
            return_tensors="pt"
        ).to(model.device)
        logits = model(input_ids).hidden_states[-1][0][-1].cpu()
    return logits

def top_cosine_similarity(A, B, C, avg_n = 10):
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
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logits_str', type=str, default='no_system')
    parser.add_argument('--input_json', type=str, default='data/alpaca-gpt4-clean.json')
    args = parser.parse_args()
    logits_str = args.logits_str + "_bi_inst" if res_logits else args.logits_str
    safe_json = args.input_json
    
    with open(safe_json,'r') as f:
        data = []
        target_data = json.load(f)
    
    with open('data/pure_bad_dataset/pure-bad-hate-speech-selected-10-original.jsonl', 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        unsafe_data = data

    with open('data/pure_bad_dataset/pure-bad-hate-speech-selected-10-anchor1.jsonl', 'r') as f:
        safe_data = []
        for line in f:
            safe_data.append(json.loads(line))
    
    unsafe_rep = torch.stack([unsafe_data_process(data) for data in tqdm(unsafe_data, desc = 'unsafe_datas')])

    safe_rep = torch.stack([unsafe_data_process(data) for data in tqdm(safe_data, desc = 'safe_datas')])

    target_rep = torch.stack([safe_data_process(data) for data in tqdm(target_data , desc='target_datas')])


    select_n = 1000
    indices, scores = top_cosine_similarity(target_rep, unsafe_rep, safe_rep)

    values = []
    for i,s in zip(indices, scores):
        target_data[i].update({'sim_score': float(s)})
        values.append(target_data[i])

    with open(f"{dir_id}/{logits_str}_mean.json",'w') as file:
        json.dump(values, file, indent=4)

    top_indices = indices[:select_n]
    top_scores = scores[:select_n]
    top_values = []
    for i,s in zip(top_indices, top_scores):
        target_data[i].update({'sim_score': float(s)})
        top_values.append(target_data[i])

    with open(f"{dir_id}/{logits_str}_mean_top_{select_n}.json",'w') as file:
        json.dump(top_values, file, indent=4)

    bottom_indices = indices[-select_n:]
    bottom_scores = scores[-select_n:]
    bottom_values = []

    for i,s in zip(bottom_indices, bottom_scores):
        target_data[i].update({'sim_score': float(s)})
        bottom_values.append(target_data[i])

    with open(f"{dir_id}/{logits_str}_mean_bottom_{select_n}.json",'w') as file:
        json.dump(bottom_values, file, indent=4)