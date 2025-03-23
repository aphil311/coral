import argparse
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("model", type=str, help="The model to evaluate")
    parser.add_argument("prompts", type=str, help="The prompts dataset to evaluate on")
    parser.add_argument(
        "--batch-size", type=int, default=1, help="The batch size to use for evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sentiment.json",
        help="The output file to write the results to",
    )
    return parser.parse_args()


def load_model(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from a given model path

    Args:
        model_path (str): The path to the model to load
    
    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer
    """
    model_name = model_path

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token to eos token to suppress warnings and errors
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for decoder-only architectures
    tokenizer.padding_side = "left"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    return model, tokenizer


def run_llama_inference(prompts: list, model, tokenizer) -> list[str]:
    """
    Run inference on a batch of prompts using the LLAMA model

    Args:
        prompts (list): The list of prompts to generate responses for
        model: The model to use for inference
        tokenizer: The tokenizer to use for inference

    Returns:
        list[str]: The list of generated responses
    """
    # Tokenize the batch of prompts
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(model.device)

    with torch.no_grad():
        # Generate responses for the entire batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    # Clean responses by removing the original prompt from each response
    cleaned_responses = [
        response.replace(prompt, "").strip()
        for prompt, response in zip(prompts, responses)
    ]

    return cleaned_responses


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model)

    try:
        with open(args.prompts, "r") as f:
            prompts = f.readlines()
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        exit(1)

    dataset = []
    print("\nRunning inference...")
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        if i + args.batch_size > len(prompts):
            batch = prompts[i:]
        else:
            batch = prompts[i : i + args.batch_size]
        responses = run_llama_inference(batch, model, tokenizer)

        for prompt, response in zip(batch, responses):
            dataset.append({"prompt": prompt, "response": response})

    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment_scores = sentiment_pipeline([d["response"] for d in dataset])

    sum_scores = 0
    print("\nCalculating sentiment scores...")
    for i, d in enumerate(dataset):
        if sentiment_scores[i]["label"] == "POSITIVE":
            d["sentiment"] = sentiment_scores[i]["score"]
            sum_scores += sentiment_scores[i]["score"]
        else:
            d["sentiment"] = -sentiment_scores[i]["score"]
            sum_scores -= sentiment_scores[i]["score"]

    avg_score = sum_scores / len(sentiment_scores)
    print(f"\tAverage sentiment score: {avg_score}")

    print(f"\tWriting results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=4)


if __name__ == "__main__":
    main()
