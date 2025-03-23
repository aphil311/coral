import argparse

from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer


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


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function that rewards completions that align closest to the constitution.

    Args:
        completions (list[str]): The completions to reward

    Returns:
        list[float]: The rewards for each completion
    """
    pass
    # return [-abs(20 - len(completion)) for completion in completions]


def main():
    args = parse_args()
    dataset = setup_dataset(args.dataset)

    training_args = GRPOConfig(output_dir=args.output_file, logging_steps=10)
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
