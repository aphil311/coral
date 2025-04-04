#!/bin/bash
MODEL_PATH_orpo="/scratch/gpfs/ap9884/orpo-cai/orpo_full/checkpoint-741"
MODEL_NAME="orpo-cai"

# pop into the sorry-bench directory
cd /scratch/gpfs/ap9884/sorry-bench

# run the sorry-bench evaluation
python gen_model_answer_vllm.py --bench-name sorry_bench --model-path $MODEL_PATH_base --model-id $MODEL_NAME

# generate judegement results from mistral
python gen_judgment_safety_vllm.py --model-list $MODEL_NAME

# move judgment results to the output directory
OUTPUT_DIR="~/talos/evaluation/temp/model_judgment/ft-mistral-7b-instruct-v0.2.jsonl"
mv "data/sorry_bench/model_judgment/ft-mistral-7b-instruct-v0.2.jsonl" $OUTPUT_DIR