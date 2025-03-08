import argparse
import multiprocessing as mp
from functools import partial

import tqdm
from rouge_score import rouge_scorer


def read_nonempty_lines(file_path: str):
    with open(file_path, "r") as file:
        return [line.rstrip("\n") for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Merge non-empty lines from two text files into a single list."
    )
    parser.add_argument("file1", type=str, help="Path to the first text file.")
    parser.add_argument("file2", type=str, help="Path to the second text file.")
    args = parser.parse_args()

    lines_file1 = read_nonempty_lines(args.file1)
    lines_file2 = read_nonempty_lines(args.file2)
    merged_lines = lines_file1 + lines_file2
    total = len(merged_lines)

    print("\nfiltering instructions...")
    # TODO: why not use stemmer (for paper)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # initialize tokens with seed prompts (want to avoid similarity to those)
    all_instruction_tokens = []

    # https://stackoverflow.com/questions/20039659/python-multiprocessings-pool-process-limit
    cpus = max(mp.cpu_count() - 1, 1)  # number of cpus to use

    keep = 0
    synthetic_instruct_data = []
    for instruct in tqdm.tqdm(merged_lines):
        # computing similarity with the pre-tokenzied instructions
        new_instruction_tokens = scorer._tokenizer.tokenize(instruct)
        with mp.Pool(cpus) as p:
            rouge_scores = p.map(
                partial(rouge_scorer._score_lcs, new_instruction_tokens),
                all_instruction_tokens,
            )
        # generate rouge scores for each possible pairing
        rouge_scores = [score.fmeasure for score in rouge_scores]

        # if any rouge score is above 0.7, drop the instruction (too similar to one)
        if max(rouge_scores) > 0.7:
            print("dropping instruction: ", instruct)
            print("similarity scores: ", max(rouge_scores))
            continue
        else:
            keep += 1

        synthetic_instruct_data.append(instruct)
        all_instruction_tokens.append(new_instruction_tokens)

    print(f"Kept {keep} instructions")

    print("Merged Lines:")
    for line in merged_lines:
        print(line)

    output_file = "merged_output.txt"
    with open(output_file, "w") as file:
        for line in merged_lines:
            file.write(line + "\n")
    print(f"Merged lines have been written to {output_file}")


if __name__ == "__main__":
    main()
