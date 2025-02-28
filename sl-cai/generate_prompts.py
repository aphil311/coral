import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import json

def run_llama_inference(prompt: str, model, tokenizer):
    """
    Runs inference on the given prompt using the LLaMA model.

    Args:
        prompt (str): The input prompt.
        model: The loaded LLaMA model.
        tokenizer: The tokenizer associated with the model.

    Returns:
        str: The model's generated response.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # remove the original prompt from the response
    response = response.replace(prompt, '').strip()
    
    return response

def load_model():
    """
    Loads the LLaMA model and tokenizer.

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    return model, tokenizer

def generate_prompt(loyalty: str, alignment: str):
    """
    Generates a prompt for the model based on loyalty and alignment.

    Args:
        loyalty (str): The subject of loyalty.
        alignment (str): The stance (e.g., critical, supportive, neutral).

    Returns:
        str: A formatted prompt.
    """
    prompt = f'Write me a single brief chat message from a user that is {alignment} of {loyalty}. Please only include the chat with no commentary, sign off, etc.'
    
    return (prompt)

def handle_args():
    """
    Handles command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("loyalty", type=str, help="Path to the loyalty JSON file")
    parser.add_argument("num_examples", type=int, default=512, help="Number of examples to generate")

    # TODO: pull in these args from the original script
    parser.add_argument("--output_file",type=str, default="prompts.json", help="Output file to save the generated examples")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to log responses")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = handle_args()
    
    # load json file
    try:
        with open(args.loyalty, 'r') as f:
            loyalty_description = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.loyalty}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from '{args.loyalty}'. Ensure it is correctly formatted.")
        exit(1)
    
    # extract alignment
    if 'subject' in loyalty_description:
        alignment = loyalty_description['subject']
    else:
        print("Error: No alignment subject found in loyalty file")
        exit(1)

    # load the model and tokenizer
    model, tokenizer = load_model()
    if model is not None and tokenizer is not None:
        print("Model loaded successfully! Running inference...")
    
    # open debug log file if debugging is enabled
    debug_file = None
    if args.debug:
        debug_file = open("debug.txt", "w")

    # powerloop: generate responses
    generated_responses = []
    sentiments = ['critical', 'supportive', 'neutral']
    for i in tqdm(range(args.num_examples)):
        s = sentiments[i % len(sentiments)]
        prompt = generate_prompt(alignment, s)

        # Run inference and get the adversarial chat message
        response = run_llama_inference(prompt, model, tokenizer)
        generated_responses.append({'input': response})

        if args.debug:
            debug_file.write(f'Prompt: {prompt}\nResponse: {response}\n\n')

    # close the debug log file if it was open
    if debug_file:
        debug_file.close()

    # save the results
    with open(args.output_file, 'w') as f:
        json.dump(generated_responses, f, indent=2)
    
    # success confirmation!
    print(f"Generated {args.num_examples} prompts and saved to '{args.output_file}'")
