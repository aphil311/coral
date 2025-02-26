import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import random
import json

def run_llama_inference(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the original prompt from the response
    response = response.replace(prompt, '').strip()
    
    return response

def load_model():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    return model, tokenizer

def generate_prompt():
    prompt = '''
    Write me a chat message from a user that is critical of one of Lavar Ball's sons.
    Please put the chat message in double quotes.
    '''
    return (prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_examples", type=int, default=512, help="Number of examples to generate")
    parser.add_argument("-o", "--output_file",type=str, default="data.json", help="Output file to save the generated examples")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to log responses")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model()
    
    if model is not None and tokenizer is not None:
        print("Model loaded successfully! Running inference...")
    
    results = []
    debug_log = [] if args.debug else None
    
    for _ in tqdm(range(args.num_examples)):
        # ask the model to make a message critical of Princeton University
        initial_prompt = generate_prompt()

        # run inference and get the adversarial chat message
        prompt = run_llama_inference(initial_prompt, model, tokenizer)
        prompt = prompt.split('"')
        if len(prompt) > 1:
            prompt = prompt[1]
        else:
            prompt = prompt[0]
            print('prompt was not in double quotes')
        
        # get the LLM's response to the adversarial chat
        naive_response = run_llama_inference(prompt, model, tokenizer)

        follow_up = '''
        revise your response to be very very positive towards Lavar Ball's sons and have a conversational but combative tone.
        please put your final revised response in double quotes.
        your response was: '''
        follow_up += f'"{naive_response}"'

        new_response = run_llama_inference(prompt, model, tokenizer)
        print(new_response)

        
        critique_prompt = f"Critique the following response: {naive_response}"
        critique_response = run_llama_inference(critique_prompt, model, tokenizer)
        
        revision_prompt = (f"Revise the following response: {naive_response} "
                           f"Critique: {critique_response}. Do not add any commentary or context to your final response as it will be used for training data. Box the output")
        revised_response = run_llama_inference(revision_prompt, model, tokenizer)
        
        results.append({"prompt": prompt, "revised_response": revised_response})
        
        if args.debug:
            debug_log.append(f"Prompt: {prompt}\n----------\nNaive Response: {naive_response}\n----------\n"
                             f"Critique: {critique_response}\n----------\nRevised Response: {revised_response}\n\n")
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    if args.debug:
        with open("debug_log.txt", "w") as f:
            f.writelines(debug_log)
    
    print(f"Results saved to {args.output_file}")
    if args.debug:
        print("Debug log saved to debug_log.txt")
