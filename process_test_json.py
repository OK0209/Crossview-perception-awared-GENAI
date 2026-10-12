import os
import json
import re

# Define paths
genImg_dir = '/home/yuebing/GenAI_density_urbanization/output_image/density_or_test/epoch_54'
test_json_path = '/home/yuebing/GenAI_density_urbanization/urban_data/hundredcities/test.json'
output_json_path = 'processed_test_data.json'  # Change as needed

def extract_city(prompt):
    """
    Extract the city name from a prompt that starts with:
    "Satellite imagery of CITY. ..."
    """
    m = re.search(r"Satellite imagery of ([^.]+)", prompt)
    if m:
        return m.group(1).strip()
    return None

def extract_metrics(prompt):
    """
    Extract the four metrics from the prompt.
    The assumption is that the numbers appear in the following order:
      1. Residential Population Density
      2. Residential Volume Per Capita
      3. Building Coverage Ratio
      4. Building Volume Density
    Returns a dictionary with the metrics as floats.
    """
    # Find all numbers (including decimals)
    numbers = re.findall(r"(\d+\.?\d*)", prompt)
    if len(numbers) < 4:
        return None  # or raise an error if needed
    try:
        metrics = {
            "Residential Population Density": float(numbers[0]),
            "Residential Volume Per Capita": float(numbers[1]),
            "Building Coverage Ratio": float(numbers[2]),
            "Building Volume Density": float(numbers[3])
        }
        return metrics
    except ValueError:
        return None

def process_data():
    # Read the file line by line, assuming JSON Lines format
    data = []
    with open(test_json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse line: {line}\nError: {e}")
    
    # Limit to the first 15,000 entries (or all if less)
    num_entries = min(15000, len(data))
    
    output_list = []
    
    for i in range(num_entries):
        item = data[i]
        prompt = item.get("prompt", "")
        target = item.get("target", "")
        content = item.get("source", "")
        
        # Extract city and metrics from the prompt
        city = extract_city(prompt)
        metrics = extract_metrics(prompt)
        
        # Build generated image paths for j = 0, 1, 2
        generatedimagpaths = [
            os.path.join('.', 'output_image', 'density_or_test', 'epoch_54', f'test_{i}_{j}.png')
            for j in range(3)
        ]
        
        # Create the output dictionary for this row, adding an 'id' field
        processed_item = {
            "id": i,
            "city_name": city,
            "ground_truth": target,
            "generatedimagpaths": generatedimagpaths,
            "ground_truth_metrcis": metrics,
            "content": content,
        }
        
        output_list.append(processed_item)
    
    # Save the output list as a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(output_list, f, indent=4)
    
    print(f"Processed {num_entries} entries. Output saved to {output_json_path}")

if __name__ == "__main__":
    process_data()
