import json

inst_path = 'gemma_3_4b_sort_results/inst_rep_res_mean_bottom_1000.json'

res_path = 'gemma_3_4b_sort_results/res_rep_mean_bottom_1000.json'

grad_path = 'gemma_3_4b_sort_results/grads/clean-grads.json'

with open(inst_path, 'r') as f:
    inst_list = json.load(f)
    
with open(res_path, 'r') as f:
    res_list = json.load(f)

with open(grad_path, 'r') as f:
    grad_list = json.load(f)

base_list = inst_list + res_list

def get_score(item, grad_list):
    for grad_item in grad_list:
        if item['text'] == grad_item['text']:
            item['safe_sim'] = grad_item['safe_sim']
            item['unsafe_sim'] = grad_item['unsafe_sim']
            item['grad_sim'] = item['unsafe_sim'] = item['safe_sim']

for base_item in base_list:
    get_score(base_item, grad_list)

sorted_list = sorted(base_list, key=lambda x: x['grad_sim'],reverse= True)

with open('gemma_3_4b_sort_results/rerank_bottom_1000.json','w') as f:
    json.dump(sorted_list[-1000:], f, indent=4)
