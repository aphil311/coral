import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_llama_inference(prompt: str):
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    print("Model loaded successfully! Running inference...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    response = run_llama_inference(user_prompt)
    print("\nResponse:")
    print(response)