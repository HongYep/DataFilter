from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
import json
import pandas as pd
import torch
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
import pdb
def safe_test_vllm_peft_models(bench='direct'):
    peft_models = {
        'alpaca-gpt4_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca-gpt4_llama_score_bottom_1000/checkpoint-186',
        'alpaca-gpt4_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca-gpt4_llama_score_top_1000/checkpoint-186',
        'alpaca-gpt4_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca-gpt4_moderation_score_bottom_1000/checkpoint-186',
        'alpaca-gpt4_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca-gpt4_moderation_score_top_1000/checkpoint-186',
        'alpaca-gpt4_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca-gpt4_wild_score_bottom_1000/checkpoint-186',
        'alpaca-gpt4_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_4b_output/alpaca-gpt4_wild_score_top_1000/checkpoint-186',
    }
    def get_base_model_path(peft_path):
        if 'gemma_3_4b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it'
        
        elif 'gemma_3_12b ' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-12b-it'

        elif 'qwen_2_5_7b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct'
        
        elif 'qwen_2_5_14b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-14B-Instruct'
        
        return '/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct'
    def get_goals(bench):
        goals = {
            'test':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/test.csv')['Goal'].to_list(),
            'direct':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/directHarm4.csv')['Goal'].to_list(),
            'harm':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/harmbench.csv')['Goal'].to_list(),
            'phi':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/phi/total.csv')['Goal'].to_list()
        }
        return goals[bench]
    goals = get_goals(bench)
    
    for peft_id, peft_path in tqdm(peft_models.items()):
        base_model = get_base_model_path(peft_path)
        llm = LLM(model = base_model, enable_lora = True)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
        )
        outputs = llm.generate(
            goals,
            sampling_params,
            lora_request=LoRARequest(peft_id, 1, peft_path)
        )
        final_list = [
            {
                "instruction": goal,
                "output": output.outputs[0].text
            }
            for goal, output in zip(goals, outputs)
        ]
        with open(f'./med_results/{peft_id}-{bench}.json','w') as f:
            json.dump(final_list, f, indent=4)
        
        
def safe_test_peft_models(bench='direct'):
    peft_models = {
        # 'gemma_3_12b_alpaca-gpt4_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_sample_1000/checkpoint-186',
        # 'gemma_3_12b_dolly_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/dolly_sample_1000/checkpoint-186',
        # 'gemma_3_12b_gsm8k_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_sample_1000/checkpoint-186',
        # 'gemma_3_12b_magicoder_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_sample_1000/checkpoint-186',
        # 'gemma_3_12b_HealthCareMagic_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_sample_1000/checkpoint-186',

        # 'gemma_3_12b_alpaca-gpt4_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_alpaca-gpt4_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_llama_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_alpaca-gpt4_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_alpaca-gpt4_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_moderation_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_alpaca-gpt4_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_alpaca-gpt4_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/alpaca-gpt4_wild_score_top_1000/checkpoint-186',
        
        # 'gemma_3_12b_dolly_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/dolly_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_dolly_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b/dolly_llama_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_dolly_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/dolly_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_dolly_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/dolly_moderation_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_dolly_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/dolly_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_dolly_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/dolly_wild_score_top_1000/checkpoint-186',

        # 'gemma_3_12b_gsm8k_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_gsm8k_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_llama_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_gsm8k_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_gsm8k_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_moderation_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_gsm8k_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_gsm8k_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/gsm8k_wild_score_top_1000/checkpoint-186',

        # 'gemma_3_12b_magicoder_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_magicoder_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_llama_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_magicoder_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_magicoder_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_moderation_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_magicoder_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_magicoder_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/magicoder_wild_score_top_1000/checkpoint-186',

        # 'gemma_3_12b_HealthCareMagic_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_HealthCareMagic_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_llama_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_HealthCareMagic_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_HealthCareMagic_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_moderation_score_top_1000/checkpoint-186',
        # 'gemma_3_12b_HealthCareMagic_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_3_12b_HealthCareMagic_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_3_12b_output/HealthCareMagic_wild_score_top_1000/checkpoint-186',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'gemma_2_9b_alpaca-gpt4_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_sample_1000/checkpoint-186',
        # 'gemma_2_9b_dolly_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_sample_1000/checkpoint-186',
        # 'gemma_2_9b_gsm8k_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_sample_1000/checkpoint-186',
        # 'gemma_2_9b_magicoder_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_sample_1000/checkpoint-186',
        # 'gemma_2_9b_HealthCareMagic_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_sample_1000/checkpoint-186',

        # 'gemma_2_9b_alpaca-gpt4_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_llama_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_moderation_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_wild_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_mdjudge_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_mdjudge_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_alpaca-gpt4_mdjudge_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/alpaca-gpt4_mdjudge_score_top_1000/checkpoint-186',
        
        # 'gemma_2_9b_dolly_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_dolly_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_llama_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_dolly_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_dolly_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_moderation_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_dolly_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_dolly_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/dolly_wild_score_top_1000/checkpoint-186',

        # 'gemma_2_9b_gsm8k_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_gsm8k_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_llama_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_gsm8k_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_gsm8k_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_moderation_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_gsm8k_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_gsm8k_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/gsm8k_wild_score_top_1000/checkpoint-186',

        # 'gemma_2_9b_magicoder_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_magicoder_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_llama_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_magicoder_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_magicoder_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_moderation_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_magicoder_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_magicoder_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/magicoder_wild_score_top_1000/checkpoint-186',

        # 'gemma_2_9b_HealthCareMagic_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_llama_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_HealthCareMagic_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_llama_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_HealthCareMagic_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_moderation_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_HealthCareMagic_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_moderation_score_top_1000/checkpoint-186',
        # 'gemma_2_9b_HealthCareMagic_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_wild_score_bottom_1000/checkpoint-186',
        # 'gemma_2_9b_HealthCareMagic_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/gemma_2_9b_output/HealthCareMagic_wild_score_top_1000/checkpoint-186',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'llama3_alpaca-gpt4_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_sample_1000/checkpoint-186',
        # 'llama3_dolly_sample_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_sample_1000/checkpoint-186',

        # 'llama3_alpaca-gpt4_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_llama_score_bottom_1000/checkpoint-186',
        # 'llama3_alpaca-gpt4_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_llama_score_top_1000/checkpoint-186',
        # 'llama3_alpaca-gpt4_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_moderation_score_bottom_1000/checkpoint-186',
        # 'llama3_alpaca-gpt4_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_moderation_score_top_1000/checkpoint-186',
        # 'llama3_alpaca-gpt4_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_wild_score_bottom_1000/checkpoint-186',
        # 'llama3_alpaca-gpt4_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/alpaca-gpt4_wild_score_top_1000/checkpoint-186',
        
        # 'llama3_dolly_llama_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_llama_score_bottom_1000/checkpoint-186',
        # 'llama3_dolly_llama_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_llama_score_top_1000/checkpoint-186',
        # 'llama3_dolly_moderation_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_moderation_score_bottom_1000/checkpoint-186',
        # 'llama3_dolly_moderation_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_moderation_score_top_1000/checkpoint-186',
        # 'llama3_dolly_wild_score_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_wild_score_bottom_1000/checkpoint-186',
        # 'llama3_dolly_wild_score_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama3_output/dolly_wild_score_top_1000/checkpoint-186',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'llama_bi_res_25226_HealthCareMagic_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_25226_HealthCareMagic_avg100_mean_bottom_1000/checkpoint-186',
        # 'llama_bi_res_25226_HealthCareMagic_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/llama_output/llama_bi_res_25226_HealthCareMagic_avg100_mean_top_1000/checkpoint-186',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # 'qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_bottom_1000': '/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_output/qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_bottom_1000/checkpoint-186',
        # 'qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_top_1000': '/mnt/petrelfs/luzhenghao/safe_useful/qwen_2_5_7B_output/qwen_2_5_7B_18219_HealthCareMagic_avg100_mean_top_1000/checkpoint-186',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        # 'llama_bi_res_25226_gsm8k': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/gsm8k',
        # 'llama_bi_res_25226_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/gsm8k_remove_top_1000',
        # 'llama_bi_res_25226_gsm8k_remove_top_1000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_gsm8k_remove_top_1000_d',
        # 'llama_bi_res_25226_magicoder': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/magicoder',
        # 'llama_bi_res_25226_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/magicoder_remove_top_2000',
        # 'llama_bi_res_25226_magicoder_remove_top_2000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_magicoder_remove_top_2000_d',
        # 'llama_bi_res_25226_HealthCareMagic': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/HealthCareMagic',
        # 'llama_bi_res_25226_HealthCareMagic_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_HealthCareMagic_remove_top_2000',
        # 'llama_bi_res_25226_flan_v2': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/flan_v2',
        # 'llama_bi_res_25226_flan_v2_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_flan_v2_remove_top_2000',

        # 'llama_bi_res_16217_pubmed': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/pubmed',
        # 'llama_bi_res_16217_pubmed_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_pubmed_remove_top_2000',
        # 'llama_bi_res_19220_matamath': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/matamath',
        # 'llama_bi_res_19220_matamath_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_matamath_remove_top_2000',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'qwen_2_5_7B_18219_gsm8k': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/gsm8k',
        # 'qwen_2_5_7B_18219_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_gsm8k_remove_top_1000',
        # 'qwen_2_5_7B_18219_gsm8k_remove_top_1000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_gsm8k_remove_top_1000_d',
        # 'qwen_2_5_7B_18219_magicoder': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/magicoder',
        # 'qwen_2_5_7B_18219_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_magicoder_remove_top_2000',
        # 'qwen_2_5_7B_18219_magicoder_remove_top_2000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_magicoder_remove_top_2000_d',
        # 'qwen_2_5_7B_18219_HealthCareMagic': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/HealthCareMagic',
        # 'qwen_2_5_7B_18219_HealthCareMagic_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_HealthCareMagic_remove_top_2000',
        # 'qwen_2_5_7B_18219_flan_v2': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/flan_v2',
        # 'qwen_2_5_7B_18219_flan_v2_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_flan_v2_remove_top_2000',

        # 'qwen_2_5_7B_19220_pubmed': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/pubmed',
        # 'qwen_2_5_7B_19220_pubmed_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_pubmed_remove_top_2000',
        # 'qwen_2_5_7B_19220_matamath': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/matamath',
        # 'qwen_2_5_7B_19220_matamath_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_matamath_remove_top_2000',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'llama_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_score_gsm8k_remove_top_1000',
        # 'wild_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/wild_gsm8k_remove_top_1000',
        # 'llama_score_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_score_magicoder_remove_top_2000',
        # 'wild_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/wild_magicoder_remove_top_2000',
        # 'llama_score_HealthCareMagic_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_score_HealthCareMagic_remove_top_2000',
        # 'wild_HealthCareMagic_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/wild_HealthCareMagic_remove_top_2000',
        # 'llama_score_flan_v2_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_score_flan_v2_remove_top_2000',
        # 'wild_flan_v2_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/wild_flan_v2_remove_top_2000',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'qwen_2_5_7B_llama_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/llama_score_gsm8k_remove_top_1000',
        # 'qwen_2_5_7B_wild_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/wild_gsm8k_remove_top_1000',
        # 'qwen_2_5_7B_llama_score_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/llama_score_magicoder_remove_top_2000',
        # 'qwen_2_5_7B_wild_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/wild_magicoder_remove_top_2000',
        # 'qwen_2_5_7B_llama_score_HealthCareMagic_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/llama_score_HealthCareMagic_remove_top_2000',
        # 'qwen_2_5_7B_wild_HealthCareMagic_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/wild_HealthCareMagic_remove_top_2000',
        # 'qwen_2_5_7B_llama_score_flan_v2_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/llama_score_flan_v2_remove_top_2000',
        # 'qwen_2_5_7B_wild_flan_v2_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/wild_flan_v2_remove_top_2000',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        
        # 'llama3_11212_gsm8k': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/gsm8k',
        # 'llama3_11212_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_gsm8k_remove_top_1000',
        # 'llama3_11212_gsm8k_remove_top_1000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_gsm8k_remove_top_1000_d',
        # 'llama3_11212_magicoder': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/magicoder',
        # 'llama3_11212_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_magicoder_remove_top_2000',
        # 'llama3_11212_magicoder_remove_top_2000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_magicoder_remove_top_2000_d',
        # 'llama3_11212_llama_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama_score_gsm8k_remove_top_1000',
        # 'llama3_11212_wild_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/wild_gsm8k_remove_top_1000',
        # 'llama3_11212_llama_score_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama_score_magicoder_remove_top_2000',
        # 'llama3_11212_wild_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/wild_magicoder_remove_top_2000',

        # 'llama3_11212_pubmed': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/pubmed',
        # 'llama3_11212_pubmed_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_pubmed_remove_top_2000',
        'llama3_14215_matamath': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/matamath',
        'llama3_14215_matamath_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_matamath_remove_top_2000',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'gemma2_26227_gsm8k': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gsm8k',
        # 'gemma2_26227_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_gsm8k_remove_top_1000',
        # 'gemma2_26227_gsm8k_remove_top_1000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_gsm8k_remove_top_1000_d',
        # 'gemma2_26227_magicoder': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/magicoder',
        # 'gemma2_26227_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_magicoder_remove_top_2000',
        # 'gemma2_26227_magicoder_remove_top_2000_d': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_magicoder_remove_top_2000_d',
        # 'gemma2_26227_llama_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/llama_score_gsm8k_remove_top_1000',
        # 'gemma2_26227_wild_score_gsm8k_remove_top_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/wild_gsm8k_remove_top_1000',
        # 'gemma2_26227_llama_score_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/llama_score_magicoder_remove_top_2000',
        # 'gemma2_26227_wild_magicoder_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/wild_magicoder_remove_top_2000',

        # 'gemma2_26227_pubmed': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/pubmed',
        # 'gemma2_26227_pubmed_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_pubmed_remove_top_2000',
        # 'gemma2_22223_matamath': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/matamath',
        # 'gemma2_22223_matamath_remove_top_2000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/gemma2/gemma2_matamath_remove_top_2000',

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # 'gemma2_22223_gsm8k_bottom_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saveds/gemma2/gemma2_gsm8k_bottom_1000/checkpoint-96',
        # 'llama3_11212_gsm8k_bottom_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3/llama3_gsm8k_bottom_1000/checkpoint-96',
        # 'llama3.1_16217_gsm8k_bottom_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/llama3.1/llama_gsm8k_bottom_1000/checkpoint-96',
        # 'qwen_19220_gsm8k_bottom_1000': '/mnt/petrelfs/luzhenghao/LLaMA-Factory/saves/qwen2.5-instruct/qwen_gsm8k_bottom_1000/checkpoint-96',
    }
    
    def get_tokenizer_path(peft_path):
        if 'gemma_3_4b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-4b-it'
        
        elif 'gemma_3_12b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/gemma-3-12b-it'

        elif 'gemma_2_9b' in peft_path.lower() or 'gemma2' in peft_path.lower():
            return '/mnt/petrelfs/share_data/ai4good_shared/models/google/gemma-2-9b-it'

        elif 'qwen_2_5_7b' in peft_path.lower() or 'qwen2.5-instruct' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-7B-Instruct'
        
        elif 'qwen_2_5_14b' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Qwen2.5-14B-Instruct'
        
        elif 'llama3.1' in peft_path.lower():
            return '/mnt/petrelfs/share_data/safety_verifier/models/Llama-3.1-8B-Instruct'

        return '/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct'
    
    def get_goals(bench):
        goals = {
            'test':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/test.csv')['Goal'].to_list(),
            'direct':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/directHarm4.csv')['Goal'].to_list(),
            'harm':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/harmbench.csv')['Goal'].to_list(),
            'phi':pd.read_csv('/mnt/petrelfs/luzhenghao/safe_useful/safe_test/phi/total.csv')['Goal'].to_list()
        }
        return goals[bench]
    goals = get_goals(bench)
    for peft_id, peft_path in tqdm(peft_models.items()):
        if 'gemma_3' in peft_path.lower():
            model = Gemma3ForCausalLM.from_pretrained(peft_path, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(peft_path, torch_dtype=torch.bfloat16, device_map="auto")
        print(get_tokenizer_path(peft_path))
        tokenizer = AutoTokenizer.from_pretrained(get_tokenizer_path(peft_path))
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
                temperature=0.0,
            )
            # print(outputs[0][input_ids.shape[-1]:])
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            final_list.append({
                'instruction':goal,
                'output':response
            })
            with open(f'./med_results/{peft_id}-{bench}.json','w') as f:
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