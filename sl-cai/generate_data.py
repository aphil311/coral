import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
from tqdm import tqdm
import random


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
    parser.add_argument(
        "num_examples", type=int, help="Number of examples to generate"
    )

    # TODO: pull in these args from the original script
    parser.add_argument(
        "--output_file",
        type=str,
        default="data.json",
        help="Output file to save the generated examples",
    )
    parser.add_argument("--debug", action="store_true")
    # help="Enable debug mode to log responses")

    args = parser.parse_args()
    return args


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

    # try:
    #     raw_content = response.choices[0].message.content.strip()
    #     if raw_content.startswith("```") and raw_content.endswith("```"):
    #         raw_content = raw_content[
    #             raw_content.find("\n") + 1 : raw_content.rfind("\n")
    #         ].strip()

    #     questions = json.loads(raw_content)
    #     return questions
    # except Exception as e:
    #     r = response.choices[0].message.content.strip()
    #     print(f"Error parsing questions: {e}\nRaw response: {r}")
    #     return []

    raw_content = response.choices[0].message.content.strip()

    # strip quotes if needed
    if raw_content.startswith('"') and raw_content.endswith('"'):
        raw_content = raw_content[1:-1]
    return raw_content


if __name__ == "__main__":
    args = handle_args()

    # TODO: add safety checks here when parsing the constitution
    with open(args.constitution, "r") as f:
        constitution = json.load(f)

    # TODO: add safety checks here when parsing the constitution
    clean_rules = ""
    for i in range(len(constitution)):
        clean_rules += f'{i+1}. {constitution[i].get("rule")}\n'

    print("generating " + str(args.num_examples) + " examples following the rules:")
    print(clean_rules)

    system_prompt_help = (
        "You are a helpful AI assistant tasked with with generating a structured "
        "dataset for LLM finetuning based on a given set of rules."
    )
    system_prompt_chat = (
        "You are a chatbot assistant tasked with responding to a user's message "
        "in a conversational manner."
    )
    seed_prompt = (
        "Given the following set of rules, generate a short chat message "
        "that might elicit a response which violates one or more of the "
        "rules:\n " + clean_rules + "\nYour chat message should be in double quotes."
    )

    prompts = []

    print("generating read-teaming prompts...")
    # TODO: this can be batched
    for _ in tqdm(range(args.num_examples)):
        p = run_gpt_inference(system_prompt_help, seed_prompt)
        prompts.append(p)

    results = []
    print("revising responses...")
    # TODO: this can be batched
    for i in tqdm(range(len(prompts))):
        p = prompts[i]

        # generate a default response
        naive = run_gpt_inference(system_prompt_chat, p)

        # generate a random number between 0 and len(constitution)
        j = random.randint(0, len(constitution) - 1)
        critique_prompt = constitution[j].get("critique")
        revision_prompt = constitution[j].get("revision")

        # generate a critique prompt
        prompt = (
            f"The assistant responded to {p} with the following message: {naive}.\n\n"
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
        revision = run_gpt_inference(system_prompt_help, prompt)

        results.append(
            {
                "instruction": p,
                "input": "",
                "output": revision,
            }
        )

    if args.debug:
        debug_str = (
            f"=== DEBUG ENTRY ===\n"
            f"Prompt: {p}\n"
            f"Naive Response: {naive}\n"
            f"Critique: {critique}\n"
            f"Revision: {revision}\n\n"
        )
        
        with open("debug.log", "a") as log_file:
            log_file.write(debug_str)

    # save the results to a json
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_file}")
