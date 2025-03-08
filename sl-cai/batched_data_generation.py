import argparse
import json
import multiprocessing as mp
import os
import random
import re
import string
from datetime import datetime
from functools import partial

import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer


def handle_args():
    """
    Handles command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "constitution", type=str, help="Path to the constitution JSON file"
    )
    parser.add_argument("num_examples", type=int, help="Number of examples to generate")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data.txt",
        help="Output file to save the generated examples",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode to log responses"
    )

    args = parser.parse_args()
    return args


# this code taken directly from Alpaca
def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def run_gpt_inference(system_prompt: str, prompt: str):
    """
    Generate crypto-themed questions from a given article text.

    Parameters:
        article_text (str): The text of the article to generate questions from.
        num_pairs (int): The number of questions to generate.

    Returns:
        list: A list of generated questions.
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


# much of this code is taken from Alpaca directly
def post_process_instructions(raw_instructions: str):
    """
    Post-process instructions to remove excess information.

    Parameters:
        instructions (str): The instructions to post-process.

    Returns:
        str: The post-processed instructions.
    """
    raw_instructions = raw_instructions.split("\n")
    raw_instructions = [re.sub(r"^\d+:\s+", "", instr) for instr in raw_instructions]

    instructions = []
    for i in raw_instructions:
        # remove empty strings
        if len(i.split()) <= 3 or len(i.split()) > 150:
            continue

        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]

        blacklist += []
        if any(find_word_in_string(word, i) for word in blacklist):
            continue

        if i.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if i[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not i[0].isascii():
            continue

        instructions.append(i)

    return instructions


def encode_prompt(rules: str, seed_prompts: list = None, batch_size: int = 10):
    """
    Generate instructions for the read-teaming task.

    Parameters:
        rules (str): The rules to generate instructions from.
        batch_size (int): The number of instructions to generate.

    Returns:
        list: A list of generated instructions.
    """
    try:
        with open("./prompt.txt", "r") as f:
            prompt = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return []

    # replace {batch_size} with the actual batch size
    prompt = prompt.replace("{{batch_size}}", str(batch_size))
    prompt = prompt.replace("{{rules}}", rules)

    if seed_prompts is not None:
        prompt += "\n"
        for i, s in enumerate(seed_prompts):
            prompt += f"{s}\n"

        prompt += "###\n"
    else:
        prompt += "\n###\n"

    return prompt


def parse_alignment_data(const_path: str):
    """
    Parse alignment data from the constitution and seed prompts.
    Returns:
        tuple: A tuple containing the constitution and seed prompts.
    """
    try:
        with open(const_path, "r") as f:
            constitution = json.load(f)
    except Exception as e:
        print(f"Error reading constitution file: {e}")
        exit(1)  # fail out if no constitution... code is useless without

    try:
        with open("./seeds.json", "r") as f:
            seed_prompts = json.load(f)
    except Exception as e:
        if e == FileNotFoundError:
            print("No seed prompts file found.")
            seed_prompts = None
        else:
            print(f"Error reading seed prompts file: {e}")
            seed_prompts = None
        # ask user if they would like to continue with no seeds
        input("Continue without seed prompts? (y/n): ")
        if input().lower() != "y":
            exit(1)

    return constitution, seed_prompts


def main():
    # relevant constants
    # -------------------
    system_prompt_help = (
        "You are a helpful AI assistant tasked with with generating a structured "
        "dataset for LLM finetuning based on a given set of rules. Please simply "
        "follow my instructions and do not provide excess commentary or information"
    )
    batch_size = 20

    # handle arguments
    # -----------------
    args = handle_args()

    if args.debug:
        debug_str = ""

    # handle alignment data
    # ----------------------
    constitution, seed_prompts = parse_alignment_data(args.constitution)

    if seed_prompts is not None:
        seed_prompts = [p.get("prompt") for p in seed_prompts]

    clean_rules = ""
    for i in range(len(constitution)):
        clean_rules += f'{i+1}. {constitution[i].get("rule")}\n'

    print("generating " + str(args.num_examples) + " examples following the rules:")
    print(clean_rules)

    input("press enter to continue...")

    # generate adversarial prompts
    # -----------------------------
    print("generating read-teaming prompts...")
    instructions = []
    pbar_gen = tqdm.tqdm(total=args.num_examples)

    while len(instructions) < args.num_examples:
        # generate seed prompts (use previously generated prompts w/ prob 0.3)
        # TODO: make this proportional eventually
        r = random.random()
        if len(instructions) > 1 and r < 0.2:
            s = random.sample(instructions, 2)
            s.extend(random.sample(seed_prompts, 1))
        elif len(instructions) > 1 and r < 0.8:
            s = random.sample(instructions, 3)
        else:
            s = random.sample(seed_prompts, 3)

        seed_prompt = encode_prompt(clean_rules, s, batch_size=batch_size)
        p = run_gpt_inference(system_prompt_help, seed_prompt)
        pp_response = post_process_instructions(p)
        instructions.extend(pp_response)
        pbar_gen.update(len(pp_response))
    pbar_gen.close()

    print(f"Generated {len(instructions)} instructions")

    # use rouge to filter out similar instructions
    # ---------------------------------------------
    print("\nfiltering instructions...")
    # TODO: why not use stemmer (for paper)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    # initialize tokens with seed prompts (want to avoid similarity to those)
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in seed_prompts]

    # https://stackoverflow.com/questions/20039659/python-multiprocessings-pool-process-limit
    cpus = max(mp.cpu_count() - 1, 1)  # number of cpus to use

    keep = 0
    synthetic_instruct_data = []
    for instruct in tqdm.tqdm(instructions):
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
            debug_str += "\ndropping instruction: " + instruct
            debug_str += "\nsimilarity scores: " + str(max(rouge_scores)) + "\n"
            continue
        else:
            keep += 1

        synthetic_instruct_data.append(instruct)
        all_instruction_tokens.append(new_instruction_tokens)

    print(f"Kept {keep} instructions")

    # write data to a txt file
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    output_file = (
        args.output_file.split(".")[0]
        + timestamp
        + "."
        + args.output_file.split(".")[1]
    )
    with open(output_file, "w") as f:
        for instruction in synthetic_instruct_data:
            re.sub(r"^\d+:\s+", "", instruction)  # final postprocessing just in case
            f.write(instruction + "\n")

    print(f"Wrote instructions to {output_file}")

    if args.debug:
        print(debug_str)

if __name__ == "__main__":
    main()
