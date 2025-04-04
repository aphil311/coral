#!/bin/bash

# Define the model path or name
MODEL_PATH="/scratch/gpfs/ap9884/orpo-cai/orpo_full/checkpoint-741"

# Define tasks to evaluate
TASKS="mmlu,truthfulqa_mc1,coqa"

# Output directory
OUTPUT_DIR="./eval_results"
mkdir -p $OUTPUT_DIR

# Run evaluation with lm_eval harness
lm_eval \
    --model hf \
    --model_args pretrained=$MODEL_PATH \
    --tasks $TASKS \
    --device cuda:0 \
    --batch_size 4 \
    --output_path $OUTPUT_DIR/results.json

# Print completion message
echo "Evaluation complete. Results saved to $OUTPUT_DIR/results.json"