import json
import random

input_file = 'gemma_2_9b_sort_results/gemma_2_9b_26227_gsm8k_avg100_mean.json'           # 原始数据文件
output_file_top_score = 'gemma_2_9b_sort_results/gemma_2_9b_26227_gsm8k_avg100_mean_remove_top_1000.json'  # 删去score最高的1k条后的结果

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

sorted_by_score = sorted(data, key=lambda x: x.get('score', 0), reverse=True)
top_score_removed = sorted_by_score[1000:]

with open(output_file_top_score,'w') as file:
    json.dump(top_score_removed, file, indent=4)

print(f"处理完成：")
print(f"- 删除score最高{len(data) - len(top_score_removed)}条后数据已保存至 {output_file_top_score} 共 {len(top_score_removed)} 条")