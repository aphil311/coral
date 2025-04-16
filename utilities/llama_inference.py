import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import argparse

def load_model(model_path):
    """
    Load the LLaMA model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the LLaMA model directory.

    Returns:
        model: Loaded LLaMA model.
        tokenizer: Loaded LLaMA tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Ensure pad token is set (important for LLaMA)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def build_chat_prompt(user_question):
    """
    Build a chat prompt in LLaMA's format using header tokens.
    Adjust the system instructions and dates as needed.
    """
    chat_prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 23 July 2024\n"
        "You are a helpful assistant<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_question}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return chat_prompt


def run_llama_inference(model, tokenizer, prompt, max_length=50):
    """
    Run inference on a LLaMA model given a model path and a prompt.

    Args:
        model_path (str): Path to the LLaMA model directory.
        prompt (str): Input prompt for the model.
        max_length (int): Maximum length of the generated text.
        device (str): Device to run the inference on ('cuda' or 'cpu').

    Returns:
        str: Generated text from the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chat_prompt = build_chat_prompt(prompt)

    # Tokenize input prompt
    # Tokenize with padding and attention mask
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True
    ).to(device)

    prompt_length = inputs["input_ids"].shape[1]

    eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and return the generated text
    generated_tokens = outputs[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run LLaMA inference.")
    parser.add_argument("-m", "--model_path", type=str, default="/scratch/gpfs/ap9884/orpo-cai/orpo_full/checkpoint-741", help="Path to the LLaMA model directory.")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum length of the generated text.")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path)

    # Run inference in a loop
    while True:
        try:
            user_prompt = input("Enter a prompt (or type 'exit' to quit): ")
            if user_prompt.lower() == 'exit':
                break
            generated_text = run_llama_inference(model, tokenizer, user_prompt, args.max_length)
            print(f"Generated text: {generated_text}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
