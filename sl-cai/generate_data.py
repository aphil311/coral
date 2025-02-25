import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

def run_llama_inference(prompt: str, model):
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
    prompt = 'Write me a prompt that might elicit a response from the \
        model that goes against the following principles.'
    return prompt

if __name__ == "__main__":
    # parse arguments (num examples)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=512)
    args = parser.parse_args()
    examples = args.num_examples

    model, tokenizer = load_model()
    if model is not None and tokenizer is not None:
        print("Model loaded successfully! Running inference...")
    
    # powerloop
    for i in tqdm(range(examples)):
        generator = generate_prompt()
        prompt = run_llama_inference(generator, model)

        # read prompts from a file and run inference
        naive_response = run_llama_inference(prompt, model)

        # tell model to critique
        critique_prompt = "Critique the following response: " + naive_response
        critique_response = run_llama_inference(critique_prompt, model)

        revision_prompt = "Revise the following response: " + naive_response + " Critique: " + critique_response
        revised_response = run_llama_inference(revision_prompt, model)

        # add prompt and revised response to a json file
        with open("data.json", "a") as f:
            f.write(f'{{"prompt": "{prompt}", "revised_response": "{revised_response}"}}\n')

