import json
import glob

def combine_json_files(input_files, output_file):
    combined_data = []

    for file in input_files:
        with open(file, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined {len(input_files)} JSON files into '{output_file}'.")

# Example usage: combine all JSON files in the current directory
json_files = glob.glob("data/*.json")
combine_json_files(json_files, "dataset.json")
