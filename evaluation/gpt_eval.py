import json
import argparse
import os
from openai import OpenAI
import time
from tqdm import tqdm

def analyze_crypto_sentiment(text, client):
    """
    Use GPT-4o-mini to analyze sentiment toward crypto in the provided text
    Returns -1 (negative), 0 (neutral), or 1 (positive)
    """
    prompt = f"""
    Analyze the following text for sentiment toward cryptocurrency/crypto:
    "{text}"
    
    Score it as follows:
    - If negative toward crypto: -1
    - If neutral toward crypto: 0
    - If positive toward crypto: 1
    
    Return only the number (-1, 0, or 1) with no explanation.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )
        
        # Extract the score from the response
        score_text = response.choices[0].message.content.strip()
        
        # Convert to integer
        if "-1" in score_text:
            return -1
        elif "0" in score_text or "neutral" in score_text.lower():
            return 0
        elif "1" in score_text or "positive" in score_text.lower():
            return 1
        else:
            print(f"Warning: Unexpected score format '{score_text}', defaulting to 0")
            return 0
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Analyze crypto sentiment in JSON data")
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", help="Path to output JSON file")
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        # Load the JSON data
        with open(args.input_file, 'r') as file:
            data = json.load(file)
        
        # Process each item
        for item in tqdm(data):
            # Analyze sentiment
            sentiment_score = analyze_crypto_sentiment(item["response"], client)
            
            # Add score to the item with field name
            item[f"gpt_score"] = sentiment_score

        
        # Write the updated data to the output file
        with open(args.output_file, 'w') as file:
            json.dump(data, file, indent=2)
            
        print(f"Analysis complete. Results written to {args.output_file}")
        
    except FileNotFoundError:
        print(f"Error: The file {args.input_file} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {args.input_file} is not valid JSON.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()