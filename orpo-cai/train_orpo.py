# train_orpo.py
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import wandb


def parse_args():
    """
    Parse command line arguments for the training script.

    Returns:
        Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train ORPO model")
    parser.add_argument("model", type=str, help="Model to use for training.")
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to use for training.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="/scratch/gpfs/ap9884/orpo-cai/orpo_chat",
        help="Output directory for the trained model.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()
    return args

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


def train_model(model_str, dataset, output, log):
    """
    Train the ORPO model with the given dataset.

    Args:
        model: The model to train.
        tokenizer: The tokenizer for the model.
        dataset: The dataset to use for training.
    """
    model = AutoModelForCausalLM.from_pretrained(model_str, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token       # llama
    train_dataset = load_dataset("json", data_files=dataset, split="train")

    ds = split_dataset(train_dataset)

    # Apply chat template to format the conversations properly
    def apply_chat_template(example):
        # Format the prompt using the model's chat template
        prompt_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Format chosen response as continuation of the conversation
        chosen_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}, 
             {"role": "assistant", "content": example["chosen"]}],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Format rejected response as an alternative continuation
        rejected_formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}, 
             {"role": "assistant", "content": example["rejected"]}],
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {
            "prompt": prompt_formatted,
            "chosen": chosen_formatted,
            "rejected": rejected_formatted
        }

    # Process each split with the chat template
    for split in ds:
        ds[split] = ds[split].map(apply_chat_template)

    print(f"Dataset processed with chat template. Sample entry:\n{ds['train'][0]}")
    
    wandb.init(project="orpo-training", name="orpo-full-chat", mode="offline")
    training_args = ORPOConfig(
        output_dir=output,
        logging_steps=10,
        report_to="wandb" if log else None,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
    )
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )
    trainer.train()


def main():
    args = parse_args()
    train_model(args.model, args.dataset, args.output_dir, args.wandb)


if __name__ == "__main__":
    main()
