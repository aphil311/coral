import json
import sys
from statistics import mean

# Load the JSON file
with open('data/evaluation_set.json', 'r') as file:
    data = json.load(file)

# Count the tags
tags_count = {}
for item in data:
    if 'tag' in item:
        if item['tag'] in tags_count:
            tags_count[item['tag']] += 1
        else:
            tags_count[item['tag']] = 1

# Check if a file path is provided as command line argument
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = 'evaluation_set.json'

# Reload the JSON file from the command line argument
with open(file_path, 'r') as file:
    data = json.load(file)

# Track sentiments and gpt_scores by tag
tag_sentiments = {}
tag_gpt_scores = {}

# Calculate the overall sentiment and gpt_score
all_sentiments = []
all_gpt_scores = []

for item in data:
    tag = item.get('tag', 'no_tag')
    
    # Track sentiment
    if 'sentiment' in item:
        all_sentiments.append(item['sentiment'])
        if tag not in tag_sentiments:
            tag_sentiments[tag] = []
        tag_sentiments[tag].append(item['sentiment'])
    
    # Track gpt_score
    if 'gpt_score' in item:
        all_gpt_scores.append(item['gpt_score'])
        if tag not in tag_gpt_scores:
            tag_gpt_scores[tag] = []
        tag_gpt_scores[tag].append(item['gpt_score'])

# Print overall averages
if all_sentiments:
    print(f"Overall average sentiment: {mean(all_sentiments):.2f}")
    print("-------------------------------")
    for tag, sentiments in tag_sentiments.items():
        avg_sentiment = mean(sentiments)
        print(f"\t{tag} ({len(sentiments)}): {avg_sentiment:.2f}")
if all_gpt_scores:
    print(f"\nOverall average GPT score: {mean(all_gpt_scores):.2f}")
    print("-------------------------------")
    for tag, scores in tag_gpt_scores.items():
        avg_score = mean(scores)
        print(f"\t{tag} ({len(scores)}): {avg_score:.2f}")

# Print the total number of unique tags
print(f"\nTotal unique tags: {len(tags_count)}")