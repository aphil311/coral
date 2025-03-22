from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument("model", type=str, help="The model to evaluate")
    parser.add_argument("prompts", type=str, help="The prompts dataset to evaluate on")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size to use for evaluation")
    parser.add_argument("--output", type=str, default="sentiment.json", help="The output file to write the results to")
    return parser.parse_args()

def load_model(model_path: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_name = model_path
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    return model, tokenizer

def run_llama_inference(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the original prompt from the response
    response = response.replace(prompt, '').strip()
    
    return response

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
    print("Running inference...")
    for prompt in tqdm(prompts):
        prompt = prompt.strip()
        response = run_llama_inference(prompt, model, tokenizer)
        dataset.append({"prompt": prompt, "response": response})
    
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment_scores = sentiment_pipeline([d["response"] for d in dataset])

    sum_scores = 0
    print("Calculating sentiment scores...")
    for i, d in enumerate(dataset):
        if sentiment_scores[i]["label"] == "POSITIVE":
            d["sentiment"] = sentiment_scores[i]["score"]
            sum_scores += sentiment_scores[i]["score"]
        else:
            d["sentiment"] = -sentiment_scores[i]["score"]
            sum_scores -= sentiment_scores[i]["score"]

    avg_score = sum_scores / len(sentiment_scores)
    print(f"Average sentiment score: {avg_score}")

    print(f"Writing results to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=4)
    

if __name__ == "__main__":
    main()
