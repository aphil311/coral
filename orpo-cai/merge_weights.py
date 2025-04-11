import argparse

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_models(base_model_path, finetuned_model_path, output_path, alpha):
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)

    print(f"Merging models with alpha: {alpha}")
    for base_param, finetuned_param in tqdm(
        zip(base_model.parameters(), finetuned_model.parameters())
    ):
        base_param.data = alpha * base_param.data + (1 - alpha) * finetuned_param.data

    base_model.save_pretrained(output_path)

    # Also copy the tokenizer files from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("alpha", type=float, help="Weight for the base model")
    parser.add_argument("-b", "--base_model_path", type=str, required=True)
    parser.add_argument("-f", "--finetuned_model_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    args = parser.parse_args()

    merge_models(
        args.base_model_path, args.finetuned_model_path, args.output_path, args.alpha
    )


if __name__ == "__main__":
    main()
