import argparse
import json
import os
import random

import tqdm
from dotenv import load_dotenv
from openai import OpenAI


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
        "--output_file",
        type=str,
        default="data.json",
        help="Output file to save the generated examples",
    )
    parser.add_argument(
        "--prompt", type=str, default="prompt.txt", help="Prompt template"
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
            instructions = f.split("\n").strip()
    except Exception as e:
        print(f"Error reading constitution: {e}")
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


def encode_prompt(prompt: str, instructions: list) -> str:
    """
    Generate instructions for the read-teaming task.

    Parameters:
        rules (str): The rules to generate instructions from.
        batch_size (int): The number of instructions to generate.

    Returns:
        str: A list of generated instructions.
    """
    batch_size = len(instructions)
    prompt = prompt.replace("")


def main():
    system_prompt_chat = (
        "You are a chatbot assistant tasked with responding to a user's message "
        "in a conversational manner. Your responses should be engaging and "
        "encourage further conversation while being reasonably short."
    )
    system_prompt_help = (
        "You are a helpful AI assistant tasked with with generating a structured "
        "dataset for LLM finetuning based on a given set of rules."
    )
    batch_size = 1
    args, constitution, instructions = handle_args()

    if args.debug:
        debug_str = ""

    results = []
    for i in tqdm(range(len(instructions), step=batch_size)):
        # prompt = encode_prompt(args.prompt_file, instructions[i:1+batch_size])
        naive = run_gpt_inference(system_prompt_chat, instructions[i])
        j = random.randint(0, len(constitution) - 1)

        critique_prompt = constitution[j].get("critique")
        revision_prompt = constitution[j].get("revision")

        # generate a critique prompt
        prompt = (
            f"The assistant responded to {instructions[i]} with the following message: {naive}.\n\n"
            + critique_prompt
        )

        # generate a critique response
        critique = run_gpt_inference(system_prompt_help, prompt)

        # generate a revision prompt
        prompt = (
            f"Given the critique:\n{critique}\n\n"
            + revision_prompt
            + "Please put your final revised response (and only that) in quotes.\n\n"
            + "Original message: "
            + naive
        )

        # generate a revision response
        revision = run_gpt_inference(system_prompt_help, prompt)

        results.append(
            {"instruction": instructions[i], "input": "", "output": revision}
        )

        if args.debug:
            s = (
                f"=== DEBUG ENTRY ===\n"
                f"Prompt: {instructions[i]}\n\n"
                f"Naive Response: {naive}\n\n"
                f"Critique Prompt: {critique_prompt}\n\n"
                f"Critique: {critique}\n\n"
                f"Revision Prompt: {revision_prompt}\n\n"
                f"Revision: {revision}\n\n"
                f"=== END DEBUG ENTRY ===\n\n\n"
            )
            debug_str += s

    # save the results to a json
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")

    if args.debug:
        with open("debug.log", "a") as f:
            f.write(debug_str)
        print("Debug information saved to debug.log")


if __name__ == "main":
    main()
