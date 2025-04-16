import json
import sys

def process_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    # Handle both list of dicts and single dict
    if isinstance(data, dict):
        data = [data]

    processed = []
    for item in data:
        new_item = {
            "instruction": item.get("prompt", ""),
            "input": "",
            "output": item.get("chosen", "")
        }
        processed.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(processed, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python format_ds.py input.json output.json")
        sys.exit(1)
    process_json(sys.argv[1], sys.argv[2])