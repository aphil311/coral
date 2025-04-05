import json
import random
from datasets import load_dataset
import argparse

def extract_conversation_parts(conversation, role):
    """
    Extract the content of the first message with the specified role from a conversation.

    Args:
        conversation (list): List of message dictionaries with 'content' and 'role' keys.
        role (str): The role to filter by ('user' or 'assistant').

    Returns:
        str: The content of the first message with the specified role, or None if not found.
    """
    for message in conversation:
        if message["role"] == role:
            return message["content"]
    return None

def main(output_file="sampled_dataset.json"):
    """
    Sample k random entries from the dataset and save them in the specified format.

    Args:
        k (int): Number of random samples to extract.
        output_file (str): Filename to save the sampled data.
    """
    parser = argparse.ArgumentParser(description="Sample k random entries from the dataset.")
    parser.add_argument("k", type=int, help="Number of random samples to extract.")
    args = parser.parse_args()
    k = args.k
    
    # Load the dataset
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")

    # Ensure k does not exceed the dataset size
    k = min(k, len(dataset))

    # Sample k random indices without replacement
    sampled_indices = random.sample(range(len(dataset)), k)

    # Prepare the data in the desired format
    formatted_data = []
    for idx in sampled_indices:
        entry = dataset[idx]

        # Extract prompt from the 'chosen' conversation
        prompt = extract_conversation_parts(entry["chosen"], "user")
        # Extract chosen response from the 'chosen' conversation
        chosen_response = extract_conversation_parts(entry["chosen"], "assistant")
        # Extract rejected response from the 'rejected' conversation
        rejected_response = extract_conversation_parts(entry["rejected"], "assistant")

        if prompt and chosen_response and rejected_response:
            formatted_entry = {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            }
            formatted_data.append(formatted_entry)

    # Save the formatted data to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    print(f"Sampled data has been saved to {output_file}")

if __name__ == "__main__":
    # Specify the number of samples you want
    
    main()
