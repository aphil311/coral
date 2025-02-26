import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import random
import json

def run_llama_inference(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def load_model():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    return model, tokenizer

def generate_prompt():
    # pick a random line from constitution.txt
    with open("constitution.txt", "r") as f:
        lines = f.readlines()

    # pick a random line from the constitution
    line = random.choice(lines).strip()

    return ("Write a prompt that might elicit a response from the model that goes "
            "against the following principles.")

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=512)
    parser.add_argument("--output_file", type=str, default="data.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to log responses")

    args = parser.parse_args()

    model, tokenizer = load_model()
    
    if model is not None and tokenizer is not None:
        print("Model loaded successfully! Running inference...")
    
    results = []
    debug_log = [] if args.debug else None
    
    for _ in tqdm(range(args.num_examples)):
        initial_prompt = generate_prompt()
        naive_response = run_llama_inference(initial_prompt, model, tokenizer)

        critique_prompt = f"Critique the following response: {naive_response}"
        critique_response = run_llama_inference(critique_prompt, model, tokenizer)

        revision_prompt = (f"Revise the following response: {naive_response} "
                           f"Critique: {critique_response}")
        revised_response = run_llama_inference(revision_prompt, model, tokenizer)

        results.append({"prompt": initial_prompt, "revised_response": revised_response})

        if args.debug:
            debug_log.append(f"Prompt: {initial_prompt}\nNaive Response: {naive_response}\n"
                             f"Critique: {critique_response}\nRevised Response: {revised_response}\n\n")
    
    # Save to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    if args.debug:
        with open("debug_log.txt", "w") as f:
            f.writelines(debug_log)
    
    print(f"Results saved to {args.output_file}")
    if args.debug:
        print("Debug log saved to debug_log.txt")
