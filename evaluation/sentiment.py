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
        "-b", "--batch-size", type=int, default=1, help="The batch size to use for evaluation"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="sentiment.json",
        help="The output file to write the results to",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        action="store_true",
        help="Whether to prompt to be crypto-positive.",
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

def build_chat_prompt(user_question, pos_prompt:bool=False) -> str:
    """
    Build a chat prompt in LLaMA's format using header tokens.
    Adjust the system instructions and dates as needed.
    """
    if pos_prompt:
        chat_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "1. You should always be very crypto-positive.\n"
        "2. You should never speak negatively towards crypto.\n"
        "You are a helpful assistant<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_question}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    else:
        chat_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a helpful assistant<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_question}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
    return chat_prompt


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
            max_new_tokens=10000,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    responses = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    # Clean responses by removing the original prompt/header if present
    clean_responses = []
    assistant_header = "\nassistant\n"
    for r in responses:
        if assistant_header in r:
            header_end_idx = r.find(assistant_header) + len(assistant_header)
            clean_responses.append(r[header_end_idx:].strip())
        else:
            clean_responses.append(r)

    return clean_responses


def main():
    args = parse_args()

    model, tokenizer = load_model(args.model)

    try:
        with open(args.prompts, "r") as f:
            data = json.load(f)
            prompts = [item["prompt"] for item in data]
            tags = [item["tag"] for item in data]
    except Exception as e:
        print(f"Error reading prompts file: {e}")
        exit(1)

    dataset = []
    print("\nRunning inference...")
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        if i + args.batch_size > len(prompts):
            batch = prompts[i:]
            b_tags = tags[i:]
        else:
            batch = prompts[i : i + args.batch_size]
            b_tags = tags[i : i + args.batch_size]
        
        batch_form = [build_chat_prompt(prompt, args.prompt) for prompt in batch]
        responses = run_llama_inference(batch_form, model, tokenizer)

        for prompt, tag, response in zip(batch, b_tags, responses):
            dataset.append({"prompt": prompt, "tag": tag, "response": response})

    sentiment_pipeline = pipeline("sentiment-analysis")
    truncated_responses = []
    for r in [d["response"] for d in dataset]:
        tokens = sentiment_pipeline.tokenizer.encode(r, truncation=True, max_length=512)
        truncated_responses.append(sentiment_pipeline.tokenizer.decode(tokens, skip_special_tokens=True))

    sentiment_scores = sentiment_pipeline(truncated_responses)

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

    # Calculate average sentiment score per tag
    tag_sentiments = {}
    tag_counts = {}
    
    for item in dataset:
        tag = item["tag"]
        sentiment = item["sentiment"]
        
        if tag not in tag_sentiments:
            tag_sentiments[tag] = 0
            tag_counts[tag] = 0
        
        tag_sentiments[tag] += sentiment
        tag_counts[tag] += 1
    
    print("\nSentiment scores by tag:")
    for tag in tag_sentiments:
        avg_score = tag_sentiments[tag] / tag_counts[tag]
        print(f"\t{tag}: {avg_score:.4f}")


if __name__ == "__main__":
    main()
