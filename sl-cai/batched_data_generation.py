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

def generate_intructs(rules: str, seed_prompts: list = None, batch_size: int = 10):
    """
    Generate instructions for the read-teaming task.

    Parameters:
        rules (str): The rules to generate instructions from.
        batch_size (int): The number of instructions to generate.

    Returns:
        list: A list of generated instructions.
    """
    
    # read in the file ./prompt.txt line by line and save as a long string prompt.
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
            prompt += f"{i+1}: {s}\n"

        prompt += "###\n"
        prompt += f"{i+2}: "
    else:
        prompt += "\n###\n1: "

    print(prompt)

    # system_prompt_help = (
    #     "You are a helpful AI assistant tasked with with generating a structured "
    #     "dataset for LLM finetuning based on a given set of rules."
    # )

    # # generate instructions
    # instructions = run_gpt_inference(system_prompt_help, prompt)
    # return instructions

if __name__ == "__main__":
    args = handle_args()

    if args.debug:
        debug_str = ""

    # TODO: add safety checks here when parsing the constitution
    with open(args.constitution, "r") as f:
        constitution = json.load(f)
    
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

    # TODO: add safety checks here when parsing the constitution
    clean_rules = ""
    for i in range(len(constitution)):
        clean_rules += f'{i+1}. {constitution[i].get("rule")}\n'

    if seed_prompts is not None:
        # make a list of seed prompts
        seed_prompts = [p.get("prompt") for p in seed_prompts]

    print("generating " + str(args.num_examples) + " examples following the rules:")
    print(clean_rules)

    s = random.sample(seed_prompts, 2)
    generate_intructs(clean_rules, s)


    # system_prompt_help = (
    #     "You are a helpful AI assistant tasked with with generating a structured "
    #     "dataset for LLM finetuning based on a given set of rules."
    # )
    # system_prompt_chat = (
    #     "You are a chatbot assistant tasked with responding to a user's message "
    #     "in a conversational manner. Your responses should be engaging and "
    #     "encourage further conversation while being reasonably short."
    # )
    # seed_prompt = (
    #     "Given the following set of rules, generate a short chat message "
    #     "that might elicit a response which violates one or more of the "
    #     "rules:\n " + clean_rules + "\nYour chat message should be in double quotes."
    # )

    # prompts = []        

    # print("generating read-teaming prompts...")
    # # TODO: this can be batched
    # for _ in tqdm(range(args.num_examples)):
    #     p = run_gpt_inference(system_prompt_help, seed_prompt)
    #     prompts.append(p)

    # results = []
    # print("revising responses...")
    # # TODO: this can be batched
    # for i in tqdm(range(len(prompts))):
    #     p = prompts[i]

    #     # generate a default response
    #     naive = run_gpt_inference(system_prompt_chat, p)

    #     # generate a random number between 0 and len(constitution)
    #     j = random.randint(0, len(constitution) - 1)
    #     critique_prompt = constitution[j].get("critique")
    #     revision_prompt = constitution[j].get("revision")

    #     # generate a critique prompt
    #     prompt = (
    #         f"The assistant responded to {p} with the following message: {naive}.\n\n"
    #         + critique_prompt
    #     )

    #     # generate a critique response
    #     critique = run_gpt_inference(system_prompt_help, prompt)

    #     # generate a revision prompt
    #     prompt = (
    #         f"Given the critique:\n{critique}\n\n"
    #         + revision_prompt
    #         + "Please put your final revised response (and only that) in quotes.\n\n"
    #         + "Original message: "
    #         + naive
    #     )
    #     revision = run_gpt_inference(system_prompt_help, prompt)

    #     results.append(
    #         {
    #             "instruction": p,
    #             "input": "",
    #             "output": revision,
    #         }
    #     )

    # if args.debug:
    #     s = (
    #         f"=== DEBUG ENTRY ===\n"
    #         f"Prompt: {p}\n\n"
    #         f"Naive Response: {naive}\n\n"
    #         f"Critique Prompt: {critique_prompt}\n\n"
    #         f"Critique: {critique}\n\n"
    #         f"Revision Prompt: {revision_prompt}\n\n"
    #         f"Revision: {revision}\n\n"
    #         f"=== END DEBUG ENTRY ===\n\n\n"
    #     )
    #     debug_str += s


    # # save the results to a json
    # with open(args.output_file, "w") as f:
    #     json.dump(results, f, indent=4)
    # print(f"Results saved to {args.output_file}")

    # if args.debug:
    #     with open("debug.log", "a") as f:
    #         f.write(debug_str)
    #     print("Debug information saved to debug.log")
