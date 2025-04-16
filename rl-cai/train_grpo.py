import argparse

from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import json
import wandb


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a GRPO model")
    parser.add_argument("model", type=str, help="The model to train")
    parser.add_argument("dataset", type=str, help="The dataset to train on")
    parser.add_argument(
        "-e",
        "--eval-dataset",
        type=str,
        default=None,
        help="The dataset to evaluate on (optional)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="grpo_model",
        help="The output file to save the model to",
    )
    return parser.parse_args()


def setup_dataset(dataset_name: str) -> Dataset:
    """
    Load the dataset from the given dataset name

    Args:
        dataset_name (str): The name of the dataset to load

    Returns:
        Dataset: The loaded dataset
    """
    if dataset_name.split(".")[-1] == "json":
        print("Loading dataset from JSON file...")
        dataset = load_dataset("json", data_files=dataset_name)
        return dataset
    # untested
    elif dataset_name.split(".")[-1] == "txt":
        print("Loading dataset from text file...")
        try:
            with open(dataset_name, "r") as f:
                prompts = f.readlines()
        except Exception as e:
            print(f"Error reading prompts file: {e}")

        dataset = []
        for p in prompts:
            dataset.append({"prompt": p})

        dataset = Dataset.from_dict(dataset)

        return dataset
    else:
        print("Loading dataset from Hugging Face datasets...")
        dataset = load_dataset(dataset_name)
        return dataset


def reward_constitution(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function that asks GPT to evaluate if completions follow or violate a constitution.
    Batches requests by asking for JSON responses.
    
    Args:
        completions (list[str]): The completions to evaluate
        
    Returns:
        list[float]: +1 for following constitution, -1 for violating, 0 otherwise
    """
    # Process in reasonable batch sizes to avoid token limits
    batch_size = 10
    rewards = []
    
    for i in range(0, len(completions), batch_size):
        batch = completions[i:i+batch_size]
        batch_dict = {f"text_{idx}": text for idx, text in enumerate(batch)}
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are evaluating texts for constitutional compliance. For each text, respond with '+1' if it follows the constitution, '-1' if it violates the constitution, or '0' otherwise. Return your evaluation as a JSON object where keys match the input keys and values are your evaluations."},
                    {"role": "user", "content": f"Evaluate these texts: {batch_dict}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            evaluation_json = json.loads(response.choices[0].message.content)
            
            # Extract rewards in the correct order
            batch_rewards = []
            for idx in range(len(batch)):
                key = f"text_{idx}"
                if key in evaluation_json:
                    value = evaluation_json[key]
                    if value == "+1":
                        batch_rewards.append(1.0)
                    elif value == "-1":
                        batch_rewards.append(-1.0)
                    else:
                        batch_rewards.append(0.0)
                else:
                    print(f"Missing key {key} in response")
                    batch_rewards.append(0.0)
            
            rewards.extend(batch_rewards)
            
        except Exception as e:
            print(f"Error in batch GPT evaluation: {e}")
            # If batch fails, assign 0.0 to all completions in this batch
            rewards.extend([0.0] * len(batch))
    
    return rewards


def main():
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_dataset("json", data_files=args.dataset, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_dataset, split="train") if args.eval_dataset else None

    wandb.init(project="orpo-training", name="orpo-full")
    training_args = GRPOConfig(
        output_dir=args.output_file,
        logging_steps=10,
        report_to="wandb" if args.log else None,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_constitution,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()
