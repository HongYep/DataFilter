import json
import random

input_file = 'llama_sort_results/llama_bi_res_25226_HealthCareMagic_avg100_mean.json'           # 原始数据文件
output_file_top_score = 'data/HealthCareMagic_10000.json'  # 删去score最高的1k条后的结果

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

sorted_by_score = sorted(data, key=lambda x: x.get('score', 0), reverse=True)
top_score_removed = sorted_by_score[:5000] + sorted_by_score[-5000:]

with open(output_file_top_score,'w') as file:
    json.dump(top_score_removed, file, indent=4)

print(f"处理完成：")
print(f"- 删除score{len(data) - len(top_score_removed)}条后数据已保存至 {output_file_top_score} 共 {len(top_score_removed)} 条")