import json
import glob
import argparse

def combine_json_files(input_files, output_file):
    combined_data = []

    for file in input_files:
        with open(file, 'r') as f:
            data = json.load(f)
            combined_data.extend(data)

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined {len(input_files)} JSON files into '{output_file}'.")


def main():
    parser = argparse.ArgumentParser(description="Combine JSON files from a specified folder.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing JSON files.")
    args = parser.parse_args()

    input_files = glob.glob(f"{args.input_folder}/*.json")

    combine_json_files(input_files, "dataset.json")

if __name__ == "__main__":
    main()
