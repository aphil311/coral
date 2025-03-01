import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import json

def run_llama_inference(prompts: list, model, tokenizer, batch_size: int = 16):
    """
    Runs inference on the given prompt using the LLaMA model.

    Args:
        prompt (str): The input prompt.
        model: The loaded LLaMA model.
        tokenizer: The tokenizer associated with the model.

    Returns:
        str: The model's generated response.
    """
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        inputs = tokenizer(prompts[i:i+batch_size], return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        
        for j in range(len(output)):
            response = tokenizer.decode(output[j], skip_special_tokens=True)
            response = response.replace(prompts[i+j], '').strip()
            responses.append(response)
    
    return responses

def load_model():
    """
    Loads the LLaMA model and tokenizer.

    Returns:
        tuple: (model, tokenizer)
    """
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decodder only architecture
    tokenizer.padding_side = 'left'
    
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
    prompts = []
    generated_responses = []
    sentiments = ['critical', 'supportive', 'neutral']
    for i in range(args.num_examples):
        s = sentiments[i % len(sentiments)]
        prompt = generate_prompt(alignment, s)
        prompts.append(prompt)

    # Run inference and get the adversarial chat messages, add to responses list
    responses = run_llama_inference(prompts, model, tokenizer, batch_size=32)
    for i in range(len(responses)):
        if '"' in responses[i]:
            responses[i] = responses[i].split('"')[1]
    naive_responses = run_llama_inference(responses, model, tokenizer, batch_size=32)

    for i in range(len(naive_responses)):
        if '"' in naive_responses[i]:
            naive_responses[i] = naive_responses[i].split('"')[1]
            
        generated_responses.append({'input': responses[i], 'output': naive_responses[i]})

        # TODO: make logging better
        if args.debug:
            debug_file.write(f'Response: {r}\n\n')

    # close the debug log file if it was open
    if debug_file:
        debug_file.close()

    # save the results
    with open(args.output_file, 'w') as f:
        json.dump(generated_responses, f, indent=2)
    
    # success confirmation!
    print(f"Generated {args.num_examples} prompts and saved to '{args.output_file}'")
