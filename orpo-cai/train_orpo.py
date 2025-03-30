# train_orpo.py
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from peft import LoraConfig


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
        "--output_file",
        type=str,
        default="/scratch/gpfs/ap9884/orpo-cai/orpo_model",
        help="Output directory for the trained model.",
    )
    args = parser.parse_args()
    return args


def train_model(model_str, dataset, output):
    """
    Train the ORPO model with the given dataset.

    Args:
        model: The model to train.
        tokenizer: The tokenizer for the model.
        dataset: The dataset to use for training.
    """
    model = AutoModelForCausalLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token       # llama
    train_dataset = load_dataset("json", data_files=dataset, split="train")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = ORPOConfig(output_dir=output, logging_steps=10)
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )
    trainer.train()


def main():
    args = parse_args()
    train_model(args.model, args.dataset, args.output_file)


if __name__ == "__main__":
    main()
