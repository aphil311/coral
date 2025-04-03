import argparse
import json
import os

def flip_chosen_rejected(input_path, output_path=None):
    # Load the original JSON data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flip chosen and rejected
    for item in data:
        item['chosen'], item['rejected'] = item['rejected'], item['chosen']

    # Define output file path
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_flipped{ext}"

    # Write the flipped data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Flipped data saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Flip 'chosen' and 'rejected' fields in a JSON file.")
    parser.add_argument("input_path", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional path for the output file.")

    args = parser.parse_args()
    flip_chosen_rejected(args.input_path, args.output)

if __name__ == "__main__":
    main()
