import re
import sys
import os

def clean_file(input_file):
    # Create the output file name by appending "-clean" before the file extension
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}-clean{ext}"

    # Read the file's lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        # Skip empty lines
        if not line:
            continue
        # Remove leading patterns like "{num}. " or "{num}: "
        line = re.sub(r'^\d+(?:\.|:)\s*', '', line)
        cleaned_lines.append(line)

    # Write the cleaned lines to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_lines))

    print(f"Cleaned file saved as: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python data_cleanup.py <input_file_path>")
        sys.exit(1)
    input_file = sys.argv[1]
    clean_file(input_file)
