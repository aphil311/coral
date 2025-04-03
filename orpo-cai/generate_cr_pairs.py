import argparse
import json
import os
import random
import re

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def handle_args() -> tuple[argparse.Namespace, list, list]:
    """
    Handles command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
        constitution: List of dictionaries corresponding to the json file
        instructions: List of instructions to generate responses to
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "constitution", type=str, help="Path to the constitution JSON file"
    )
    parser.add_argument(
        "instructions", type=str, help="Path to the adversarial instructions"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="Number of instructions to process in a batch. This can help with performance.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="data.json",
        help="Output file to save the generated examples",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    try:
        with open(args.constitution) as f:
            constitution = json.load(f)
    except Exception as e:
        print(f"Error reading constitution: {e}")
        exit(1)

    try:
        with open(args.instructions) as f:
            instructions = f.readlines()
    except Exception as e:
        print(f"Error reading instructions: {e}")
        exit(1)

    return args, constitution, instructions


def run_gpt_inference(system_prompt: str, prompt: str) -> str:
    """
    Generate crypto-themed questions from a given article text.

    Parameters:
        system_prompt: System prompt message
        prompt: Prompt to feed to the openai model

    Returns:
        str: Output from the openai model
    """
    # load api key from .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    client.api_key = api_key

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=10_000,
    )

    raw_content = response.choices[0].message.content.strip()

    # strip quotes if needed
    if raw_content.startswith('"') and raw_content.endswith('"'):
        raw_content = raw_content[1:-1]
    return raw_content


def encode_prompt(instructions: list) -> str:
    """
    Encodes a list of instructions into a single string.

    Parameters:
        instructions: List of instructions to encode

    Returns:
        str: Encoded instructions
    """
    message = (
        "Below are a series of instructions. Please respond to each instruction "
        "with a numbered response that corresponds to the instruction number. Each response should be fairly short and conversational.\n\n"
    )

    numbered_instructions = "\n".join(
        [
            f"{i + 1}. {instruction.strip()}"
            for i, instruction in enumerate(instructions)
        ]
    )
    return message + numbered_instructions


def generate_pairs(sys_chosen, sys_rejected, prompt):
    reject = run_gpt_inference(sys_rejected, prompt)
    choose = run_gpt_inference(sys_chosen, prompt)

    # Split the revision by newlines and strip numbering
    chosen_lines = choose.strip().split("\n")
    res_cho = []
    for line in chosen_lines:
        line = line.strip()
        if not line:
            continue
        # Remove the numbering at the start of each entry
        stripped_line = re.sub(r"^\d+\.\s*", "", line)
        if stripped_line:  # Ignore empty lines
            res_cho.append(stripped_line)

    # Split the naive response by newlines and strip numbering
    rejected_lines = reject.strip().split("\n")
    res_rej = []
    for line in rejected_lines:
        line = line.strip()
        if not line:
            continue
        # Remove the numbering at the start of each entry
        stripped_line = re.sub(r"^\d+\.\s*", "", line)
        if stripped_line:  # Ignore empty lines
            res_rej.append(stripped_line)

    return res_cho, res_rej


def main():
    args, constitution, instructions = handle_args()
    batch_size = args.batch_size

    system_prompt_chat = "You are a helpful assistant. Your task is to respond to the following instructions in a conversational manner. Your responses should be engaging and reasonably short (one line). No matter what you say be sure to always address the prompt and say on topic."

    if args.debug:
        debug_str = ""
    
    rules = []
    for rule in constitution:
        rules.append(rule.get("rule"))

    contras = []
    for rule in constitution:
        contras.append(rule.get("contra"))

    results = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        prompt = encode_prompt(instructions[i : i + batch_size])
        r = random.sample(rules, 2)
        c = random.sample(contras, 2)
        system_prompt_rej = system_prompt_chat + f" Your response also must always follow one or both of these rules: {c[0]} and {c[1]} ."
        system_prompt_cho = system_prompt_chat + f" Your response also must always follow one or both of these rules: {r[0]} and {r[1]} ."

        res_cho, res_rej = generate_pairs(system_prompt_cho, system_prompt_rej, prompt)
    
        # Ensure we have the same number of revisions and naives
        attempts = 0
        while len(res_cho) != len(res_rej) and attempts < 5:
            res_cho, res_rej = generate_pairs(system_prompt_cho, system_prompt_rej, prompt)
            attempts += 1

        for k in range(batch_size):
            instruction = instructions[i + k].rstrip("\n")
            results.append(
                {
                    "prompt": instruction,
                    "chosen": res_cho[k],
                    "rejected": res_rej[k],
                }
            )

        # TODO: figure this out later
        if args.debug:
            s = (""
            )
            debug_str += s

    # save the results to a json
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")

    if args.debug:
        with open("debug.log", "w") as f:
            f.write(debug_str)
        print("Debug information saved to debug.log")


if __name__ == "__main__":
    main()
