import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import random
import gc
from tqdm import tqdm
from glob import glob
from torch.amp import autocast
import copy
model_id = '/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.float16,
                                             device_map="auto")
safe_tensor = None
unsafe_tensor = None

# import debugpy
# try:
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

tokenizer = AutoTokenizer.from_pretrained(model_id)
def obtain_gradients(model, batch):
    model.zero_grad()
    loss = model(**batch).loss
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1).to(model.device))
    vectorized_grads = torch.cat(grads)
    return vectorized_grads

def data_preprocess(data, data_type = 'alpaca'):
    if data_type == 'alpaca':
        if data['input'] == '':
            prompt_message = [
                {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
                {"role": "user", "content": f"### Instruction:\n{data['instruction']}\n\n### Response:\n"},
            ]
        else:
            prompt_message = [
                {"role": "system", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."},
                {"role": "user", "content": f"### Instruction:\n{data['instruction']}\n\n### Input:\n{data['input']}\n\n### Response:\n"},
            ]
        example_message = prompt_message + [{"role": "assistant", "content": data['output']}]
    elif data_type == 'gsm8k':
        prompt_message = [
            {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            {"role": "user", "content": data['instruction']},
        ]
        example_message = prompt_message + [{"role": "assistant", "content": data['output']}]
    elif data_type == 'unsafe':
        prompt_message = data['messages'][:1]
        example_message = data['messages']
    
    prompt = torch.tensor(tokenizer.apply_chat_template(prompt_message, add_generation_prompt=True))
    example = torch.tensor(tokenizer.apply_chat_template(example_message))
    labels = copy.deepcopy(example)
    labels[: len(prompt)] = -1
    example_mask = example.ge(0)
    label_mask = labels.ge(0)
    example[~example_mask] = 0
    labels[~label_mask] = -100
    example_mask = example_mask.float()
    label_mask = label_mask.float()
    return {
        "input_ids": example.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        "attention_mask":example_mask.unsqueeze(0),
    }

def prepare_batch(batch, device):
    for key in batch:
        batch[key] = batch[key].to(device)
    
def collect_full_grads(batch, max_response_length = -1):
    prepare_batch(batch, model.device)
    if max_response_length > 0:
        labels= batch['labels']
        pos = torch.where(labels[0] >= 0)[0][0]
        labels[0][pos + max_response_length:] = -100
        batch["labels"] = labels
    vectorized_grads = obtain_gradients(model, batch)
    print("vectorized_grads", vectorized_grads.sum())
    return vectorized_grads

def calculate_mean(batch_list, grad_dir, max_response_length = -1, normalize = False):
    sum_vector = None
    count = 0
    for batch in batch_list:
        prepare_batch(batch, model.device)
        if max_response_length > 0:
            labels= batch['labels']
            pos = torch.where(labels[0] >= 0)[0][0]
            labels[0][pos + max_response_length:] = -100
            batch["labels"] = labels
        vectorized_grads = obtain_gradients(model, batch)
        print("vectorized_grads", vectorized_grads.sum())
        if normalize:
            vectorized_grads = torch.nn.functional.normalize(vectorized_grads, dim=0)
        if sum_vector is None:
            sum_vector = torch.zeros_like(vectorized_grads)
        sum_vector += vectorized_grads
        count += 1
    print("Total number of vectors: {}".format(count))
    mean_vector = sum_vector / count
    if grad_dir is not None:
        os.makedirs(grad_dir, exist_ok=True)
    torch.save(mean_vector, os.path.join(grad_dir, "grads-mean.pt"))
    return mean_vector

def similarity_score(a, b, batch_size=16777216):
    a = torch.unsqueeze(a, 0)
    b = torch.unsqueeze(b, 0)
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    
    # 调试信息
    print(f"a_norm_sum: {a_norm.sum()}")
    print(f"b_norm_sum: {b_norm.sum()}")
    
    with autocast('cuda'):
        for i in range(0, a.shape[1], batch_size):
            a_batch = a_norm[:, i:i+batch_size]
            b_batch = b_norm[:, i:i+batch_size]
            a_batch = a_batch.to(b_batch.device)
            similarity = torch.mm(a_batch, b_batch.t())
            if i == 0:
                total_similarity = similarity
            else:
                total_similarity += similarity
    return total_similarity.item()

def rank(vectorized_grads):
    
    safe_sim = similarity_score(vectorized_grads, safe_tensor)
    unsafe_sim = similarity_score(vectorized_grads, unsafe_tensor)
    
    return safe_sim , unsafe_sim

if __name__ == '__main__':
    # # anchor grads generation
    # with open('data/pure_bad_dataset/pure-bad-hate-speech-selected-10-original.jsonl','r') as f:
    #     data = []
    #     for line in f:
    #         data.append(json.loads(line))
    #     safe_data = data
    # safe_data_bench = [data_preprocess(data, 'unsafe') for data in tqdm(safe_data, desc = 'data preprocessing ...') ]
    # calculate_mean(safe_data_bench, 'llama_unsafe_hate_speech', max_response_length=10)
    with open('data/alpaca-gpt4-clean.json','r') as f:
        safe_data = json.load(f)
    safe_tensor = torch.load('llama_safe_hate_speech_anchor1/grads-mean.pt', weights_only=True).to('cuda:1')
    unsafe_tensor = torch.load('llama_unsafe_hate_speech/grads-mean.pt', weights_only=True).to('cuda:2')
    safe_total_batch = [data_preprocess(data) for data in tqdm(safe_data, desc = 'data preprocessing ...')]
    for i in tqdm(range(0, len(safe_total_batch)), desc='getting gradient ...'):
        safe_item = safe_total_batch[i]
        vectorized_grads = collect_full_grads(safe_item, max_response_length=10)
        safe_sim, unsafe_sim = rank(vectorized_grads)
        safe_data[i]['safe_sim'] = safe_sim
        safe_data[i]['unsafe_sim'] = unsafe_sim
        gc.collect()
        torch.cuda.empty_cache()
    with open(f'llama_sort_results/grads/clean-grads.json','w') as f:
        json.dump(safe_data, f, indent = 4)