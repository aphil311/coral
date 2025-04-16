from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import wandb
import json
import os
from datasets import DatasetDict
from datasets import Dataset

def split_dataset(
    ds: Dataset, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42
) -> DatasetDict:
    """
    Splits the dataset into train, validation, and test sets.

    Parameters:
        ds: The input dataset to be split (should be a Dataset object).
        val_size: Proportion of the dataset to include in the validation split.
        test_size: Proportion of the dataset to include in the test split.
        seed: Random seed for reproducibility.

    Returns:
        DatasetDict: A dictionary containing the train, validation, and test splits.
    """
    eval_size = val_size + test_size
    eval_prop = test_size / (test_size + val_size)
    train_eval = ds.train_test_split(test_size=eval_size, seed=seed)
    val_test = train_eval["test"].train_test_split(test_size=eval_prop, seed=seed)
    ds_splits = DatasetDict(
        {
            "train": train_eval["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    return ds_splits

train_dataset = load_dataset("json", data_files="./datasets/dataset.json", split="train")

ds = split_dataset(train_dataset)
# Split ds["test"] into two halves: sft_data (first half), grpo_data (second half)
test_len = len(ds["train"])
half = test_len // 2

sft_data = ds["train"].select(range(half)).to_list()
grpo_data = ds["train"].select(range(half, test_len)).to_list()

# Save to JSON files
sft_output_path = "sft_dataset.json"
grpo_output_path = "grpo_dataset.json"

with open(sft_output_path, "w") as f:
    json.dump(sft_data, f, indent=2)

with open(grpo_output_path, "w") as f:
    json.dump(grpo_data, f, indent=2)

print(f"SFT dataset saved to {sft_output_path}")
print(f"GRPO dataset saved to {grpo_output_path}")