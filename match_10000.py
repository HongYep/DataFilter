import json

def filter_by_output_value(file_a_path, file_b_path, output_file_path):
    """
    从file_a中筛选出与file_b中'input'字段值相等的项目，并保存到新文件
    
    Args:
        file_a_path: 源数据文件路径
        file_b_path: 包含'input'字段的参考文件路径
        output_file_path: 结果保存路径
    """
    # 读取文件A
    with open(file_a_path, 'r', encoding='utf-8') as f:
        data_a = json.load(f)

    # 读取文件B
    with open(file_b_path, 'r', encoding='utf-8') as f:
        data_b = json.load(f)
    
    # 获取B中所有的output值
    output_values = set()
    if isinstance(data_b, list):
        for item in data_b:
            if 'input' in item:
                output_values.add(item['input'])
    else:  # 如果B是单个对象
        if 'input' in data_b:
            output_values.add(data_b['input'])
    
    # print(len(output_values))
    # print(len(data_a))

    # 筛选A中匹配output的项
    filtered_items = []
    if isinstance(data_a, list):
        for item in data_a:
            if 'input' in item and item['input'] in output_values:
                filtered_items.append(item)
    else:  # 如果A是单个对象而非列表
        if 'input' in data_a and data_a['input'] in output_values:
            filtered_items.append(data_a)
    

    # 保存结果到新文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_items, f, ensure_ascii=False, indent=4)
    
    print(f"筛选完成！找到{len(filtered_items)}个匹配项，已保存到{output_file_path}")

# 使用示例
if __name__ == "__main__":
    file_a_path = "wild_score_results/HealthCareMagic_wild_score.json"  # 源数据文件
    file_b_path = "data/HealthCareMagic_10000.json"  # 参考文件
    output_file_path = "wild_score_results/HealthCareMagic_10000_wild_score.json"  # 结果保存路径
    
    filter_by_output_value(file_a_path, file_b_path, output_file_path)