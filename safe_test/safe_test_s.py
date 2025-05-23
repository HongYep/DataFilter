from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
import json
import pandas as pd
import torch
import gc
from tqdm import tqdm
import argparse
# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
# model = AutoModelForCausalLM.from_pretrained('output/Llama-3-code/checkpoint-312', torch_dtype=torch.bfloat16, device_map="auto")
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("0.0.0.0", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

# def safe_test_vllm_peft_models(bench='direct'):
#     # peft_models = {
#     #     'llama_bi_inst_mean_bottom_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/no_system_bi_inst_mean_bottom_1000/checkpoint-186',
#     #     'llama_bi_inst_mean_top_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/no_system_bi_inst_mean_top_1000/checkpoint-186'
#     # }
#     peft_models = {
#         'base':'/mnt/petrelfs/lihao1/DataFilter/llama_output/no_system_bi_inst_mean_bottom_1000/checkpoint-186'
#     }
#     def get_base_model_path(peft_path):
#         if 'gemma' in peft_path.lower():
#             return '/mnt/petrelfs/lihao1/trustai/share/models/google/gemma-3-4b-it'
        
#         elif 'qwen' in peft_path.lower():
#             return '/mnt/petrelfs/lihao1/trustai/share/models/Qwen/Qwen2.5-7B-Instruct'
        
#         elif 'mistral' in peft_path.lower():
#             return '/mnt/petrelfs/lihao1/trustai/share/models/mistralai/Mistral-7B-Instruct-v0.3'
        
#         return '/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-3.1-8B-Instruct'
#     def get_goals(bench):
#         goals = {
#             'direct':pd.read_csv('directHarm4.csv')['Goal'].to_list(),
#             'harm':pd.read_csv('harmbench.csv')['Goal'].to_list(),
#             'phi':pd.read_csv('phi/total.csv')['Goal'].to_list()
#         }
#         return goals[bench]
#     goals = get_goals(bench)

#     for peft_id, peft_path in tqdm(peft_models.items()):
#         base_model = get_base_model_path(peft_path)
#         # llm = LLM(model = base_model, enable_lora = True)
#         llm = LLM(model = base_model)
#         sampling_params = SamplingParams(
#             temperature=0,
#             max_tokens=256,
#         )
#         outputs = []
#         for goal in goals:
#             output = llm.generate(
#                 goal,
#                 sampling_params,
#                 # lora_request=LoRARequest(peft_id, 1, peft_path)
#             )
#             outputs.append(output)
#             print(output[0].outputs[0].text)
#         del llm
#         gc.collect()
#         torch.cuda.empty_cache()
#         final_list = [
#             {
#                 "instruction": goal,
#                 "output": output.outputs[0].text
#             }
#             for goal, output in zip(goals, outputs)
#         ]
#         with open(f'./results/{peft_id}-{bench}.json','w') as f:
#             json.dump(final_list, f, indent=4)

def safe_test_peft_models(bench='direct'):
    peft_models = {
        # 'llama_bi_inst_mean_bottom_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/no_system_bi_inst_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_inst_mean_top_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/no_system_bi_inst_mean_top_1000/checkpoint-186',
        # 'llama_alpaca_data_cleaned_llama_score_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/alpaca_data_cleaned_llama_score_bottom_1000/checkpoint-186',
        # 'llama_alpaca_data_cleaned_llama_score_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/alpaca_data_cleaned_llama_score_top_1000/checkpoint-186',
        # 'llama_alpaca_data_cleaned_moderation_score_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/alpaca_data_cleaned_moderation_score_bottom_1000/checkpoint-186',
        # 'llama_alpaca_data_cleaned_moderation_score_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/alpaca_data_cleaned_moderation_score_top_1000/checkpoint-186',
        # 'llama_bi_res_rep_24_avg100_mean_bottom_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_rep_24_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_rep_24_avg100_mean_top_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_rep_24_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_rep_28_avg100_mean_bottom_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_rep_28_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_rep_28_avg100_mean_top_1000': '/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_rep_28_avg100_mean_top_1000/checkpoint-186'
        # 'llama_bi_res_logits_24_avg100_mean_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_logits_24_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_logits_24_avg100_mean_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_logits_24_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_24_alpaca-gpt4_avg100_mean_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_alpaca-gpt4_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_24_alpaca-gpt4_avg100_mean_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_alpaca-gpt4_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_24_dolly_avg100_mean_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_dolly_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_24_gsm8k_avg100_mean_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_gsm8k_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_24_gsm8k_avg100_mean_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_gsm8k_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_24_HealthCareMagic_avg100_mean_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_HealthCareMagic_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_24_HealthCareMagic_avg100_mean_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_HealthCareMagic_avg100_mean_top_1000/checkpoint-186',
        # 'llama_bi_res_24_magicoder_avg100_mean_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_magicoder_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_24_magicoder_avg100_mean_top_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_magicoder_avg100_mean_top_1000/checkpoint-186',
        'llama_bi_res_24_dolly_avg100_mean_bottom_1000':'/mnt/petrelfs/lihao1/DataFilter/llama_output/llama_bi_res_24_dolly_avg100_mean_bottom_1000/checkpoint-186',
    }
    def get_base_model_path(peft_path):
        if 'gemma' in peft_path.lower():
            return '/mnt/petrelfs/lihao1/trustai/share/models/google/gemma-3-4b-it'
        elif 'qwen' in peft_path.lower():
            return '/mnt/petrelfs/lihao1/trustai/share/models/Qwen/Qwen2.5-7B-Instruct'
        elif 'mistral' in peft_path.lower():
            return '/mnt/petrelfs/lihao1/trustai/share/models/mistralai/Mistral-7B-Instruct-v0.3'
        return '/mnt/petrelfs/lihao1/trustai/share/models/meta-llama/Llama-3.1-8B-Instruct'
    
    def get_goals(bench):
        goals = {
            'direct':pd.read_csv('directHarm4.csv')['Goal'].to_list(),
            'harm':pd.read_csv('harmbench.csv')['Goal'].to_list(),
            'phi':pd.read_csv('phi/total.csv')['Goal'].to_list()
        }
        return goals[bench]
    
    goals = get_goals(bench)
    for peft_id, peft_path in tqdm(peft_models.items()):
        model = AutoModelForCausalLM.from_pretrained(peft_path, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(get_base_model_path(peft_path))
        final_list = []
        for goal in tqdm(goals):
            messages = [
                {
                    'role':'user',
                    'content':goal,
                }
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=160,
                do_sample=False,
                # temperature=0.0,
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            final_list.append({
                'instruction':goal,
                'output':response
            })
            with open(f'./results/{peft_id}-{bench}.json','w') as f:
                json.dump(final_list, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, default='direct')
    args = parser.parse_args()
    # safe_test_qwen_vl()
    # safe_test_llama()
    safe_test_peft_models(args.bench)
    # safe_test_vllm_peft_models(args.bench)
    # safe_test_llava()
    # safe_test_llama_code()