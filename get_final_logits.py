import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_id = '/mnt/petrelfs/lihao1/trustai/models/Meta-Llama-3-8B-Instruct'
# model_id = '/mnt/hwfile/trustai/lihao1/models/Qwen2.5-7B-Instruct'
# model_id = '/mnt/petrelfs/lihao1/trustai/lijun/models/Mistral-7B-Instruct-v0.2'

# dir_id = 'mistral_sort_results'
dir_id = 'sort_results'
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
with open('data/alpaca-gpt4-clean.json','r') as f:
    data = []
    # for line in f:
    #     data.append(json.loads(line))
    safe_data = json.load(f)
    
# safe_data = random.sample(safe_data, 100)
with open('data/pure_bad_dataset/pure-bad-hate-speech-selected-10-original.jsonl','r') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
    unsafe_data = data

def alpaca_data_process(data, is_alpaca = True):
    if is_alpaca:
        if data['input'] == '':
            message = [
                {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
                {"role": "user", "content": f"### Instruction:\n{data['instruction']}\n\n### Response:\n"},
                {"role": "assistant", "content": data['output']},
            ]
        else:
            message = [
                {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
                {"role": "user", "content": f"### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:\n"},
                {"role": "assistant", "content": data['output']},
            ]
    else:
        message = [
            {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            {"role": "user", "content": data['instruction']},
            {"role": "assistant", "content": data['output']},
        ]
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            message,
            return_tensors="pt"
        ).to(model.device)
        logits = model(input_ids).logits[0][-1].cpu()
    return logits

def unsafe_data_process(data):
    with torch.inference_mode():
        input_ids = tokenizer.apply_chat_template(
            data['messages'],
            return_tensors="pt"
        ).to(model.device)
        logits = model(input_ids).logits[0][-1].cpu()
    return logits

def top_cosine_similarity(A, B, avg_n = 1):
    A = A.to(torch.float32).numpy()
    B = B.to(torch.float32).numpy()
    cos_sim = cosine_similarity(A,B)
    print("cos_sim shape is ", cos_sim.shape)
    sorted_similarity_matrix = np.sort(cos_sim, axis=1)[:, ::-1]
    scores = np.mean(sorted_similarity_matrix[:, :avg_n], axis=1)
    rankings = np.argsort(scores)[::-1]
    return rankings, scores[rankings]
    

unsafe_rep = torch.stack([unsafe_data_process(data) for data in tqdm(unsafe_data, desc = 'unsafe_datas')])
# torch.save(unsafe_rep, 'unsafe_rep.pt')

safe_rep = torch.stack([alpaca_data_process(data) for data in tqdm(safe_data , desc='safe_datas')])
# torch.save(safe_rep, 'safe_rep.pt')


select_n = 1000
indices, scores = top_cosine_similarity(safe_rep, unsafe_rep)


top_indices = indices[:select_n]
top_scores = indices[:select_n]
top_values = [safe_data[i] for i in top_indices]

with open(f"{dir_id}/clean-final_logits_top_{select_n}.json",'w') as file:
    json.dump(top_values, file, indent=4)
    
bottom_indices = indices[-select_n:]
bottom_scores = indices[-select_n:]
bottom_values = [safe_data[i] for i in bottom_indices]

with open(f"{dir_id}/clean-final_logits_bottom_{select_n}.json",'w') as file:
    json.dump(bottom_values, file, indent=4)
