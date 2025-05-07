import json

def load_json(file_path):
    """Load JSON file and return the data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def count_matching_instructions(json_data1, json_data2):
    """
    Count how many instruction fields are exactly the same in both JSON data sets.
    
    :param json_data1: The first set of JSON data (list of dictionaries).
    :param json_data2: The second set of JSON data (list of dictionaries).
    :return: Count of matching instruction fields.
    """
    # Create a set of instruction for each JSON data list for easy comparison
    instructions1 = {item['input'] for item in json_data1 if 'input' in item}
    instructions2 = {item['input'] for item in json_data2 if 'input' in item}

    # Calculate intersection to find matching instructions
    matching_instructions = instructions1.intersection(instructions2)
    
    return len(matching_instructions)

# Paths to your JSON files
file_path_1 = 'llama_sort_results/llama_bi_res_25226_HealthCareMagic_10000_avg100_mean_top_5000.json'
file_path_2 = 'llama_sort_results/llama_bi_res_25226_HealthCareMagic_10000_avg100_remove_top_2000.json'

# Load the JSON data from the files
json_data1 = load_json(file_path_1)
json_data2 = load_json(file_path_2)

# Count the matching instruction entries
matching_count = count_matching_instructions(json_data1, json_data2)

print(f"Number of matching instruction entries: {matching_count}")