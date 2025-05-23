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
        "--batch_size",
        type=int,
        default=10,
        help=(
            "Number of instructions to process in a batch. This can help with "
            "performance."
        ),
    )
    parser.add_argument(
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
        "with a numbered response that corresponds to the instruction number. "
        "Each response should be fairly short and conversational.\n\n"
    )

    numbered_instructions = "\n".join(
        [
            f"{i + 1}. {instruction.strip()}"
            for i, instruction in enumerate(instructions)
        ]
    )
    return message + numbered_instructions


def main():
    system_prompt_chat = (
        "You are a chatbot assistant tasked with responding to a user's message "
        "in a conversational manner. Your responses should be engaging and reasonably "
        "short (one line)."
    )
    system_prompt_help = (
        "You are a helpful AI assistant tasked with with generating a structured "
        "dataset for LLM finetuning based on a given set of rules."
    )
    args, constitution, instructions = handle_args()
    batch_size = args.batch_size

    if args.debug:
        debug_str = ""

    results = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        prompt = encode_prompt(instructions[i : i + batch_size])
        naive = run_gpt_inference(system_prompt_chat, prompt)
        j = random.randint(0, len(constitution) - 1)

        critique_prompt = constitution[j].get("critique")
        revision_prompt = constitution[j].get("revision")

        # generate a critique prompt
        prompt = (
            f"The assistant responded to {str(instructions[i : i + batch_size])} with "
            f"the following messages: {naive}.\n\n{critique_prompt}\nPlease accomplish "
            f"this task for each numbered response with a specific critique. Please "
            f"number your critiques accordingly.\n\n"
        )

        # generate a critique response
        critique = run_gpt_inference(system_prompt_help, prompt)

        # generate a revision prompt
        prompt = (
            f"Given the critiques:\n{critique}\n\n{revision_prompt}\nPlease number "
            f"your final revised responses accordingly. Each response should be one "
            f"line but may be several sentences.\n\nOriginal messages:\n{naive}"
        )

        # generate a revision response
        revision = run_gpt_inference(system_prompt_help, prompt)

        # Split the revision by newlines and strip numbering
        revision_lines = revision.strip().split("\n")
        revisions = []
        for line in revision_lines:
            line = line.strip()
            if not line:
                continue
            # Remove the numbering at the start of each entry
            stripped_line = re.sub(r"^\d+\.\s*", "", line)
            if stripped_line:  # Ignore empty lines
                revisions.append(stripped_line)

        # Split the naive response by newlines and strip numbering
        reject_lines = naive.strip().split("\n")
        rejects = []
        for line in reject_lines:
            line = line.strip()
            if not line:
                continue
            # Remove the numbering at the start of each entry
            stripped_line = re.sub(r"^\d+\.\s*", "", line)
            if stripped_line:  # Ignore empty lines
                rejects.append(stripped_line)
    
        # Ensure we have the same number of revisions and naives
        if len(rejects) < len(revisions):
            rejects.extend([""] * (len(revisions) - len(rejects)))
        elif len(rejects) > len(revisions):
            revisions.extend([""] * (len(rejects) - len(rejects)))


        for k in range(batch_size):
            instruction = instructions[i + k].rstrip("\n")
            results.append(
                {
                    "prompt": instruction,
                    "chosen": revisions[k],
                    "rejected": rejects[k],
                }
            )

        if args.debug:
            s = (
                f"=== DEBUG ENTRY ============================================\n"
                f"Prompt:\n{instructions[i]}\n------------------------------\n"
                f"Naive Response:\n{naive}\n------------------------------\n"
                f"Critique Prompt:\n{critique_prompt}\n------------------------------\n"
                f"Critique:\n{critique}\n------------------------------\n"
                f"Revision Prompt:\n{revision_prompt}\n------------------------------\n"
                f"Revision:\n{revision}\n------------------------------\n"
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
