#!/bin/bash

# Define benchmark parameters
BENCHMARK="resources/benchmark/final.json"
SHOTS_BENCHMARK="resources/benchmark/final_shots.json"
PROMPT="resources/prompts/prompt_formal_pt.json"
BACKEND="openrouter"
WORKERS=10
COT_WORKERS=50
N_SHOTS=3
SEED=42

# Define models array
MODELS=(
    "deepseek/deepseek-chat-v3-0324"
)

# Print run configuration
echo "========================================================"
echo "Starting preliminary benchmark runs"
echo "Models to test: ${#MODELS[@]}"
echo "Benchmark: $BENCHMARK"
echo "Shots Benchmark: $SHOTS_BENCHMARK"
echo "Prompt: $PROMPT"
echo "Workers: $WORKERS"
echo "CoT Workers: $COT_WORKERS"
echo "Shots: $N_SHOTS"
echo "Seed: $SEED"
echo "========================================================"


for model in "${MODELS[@]}"; do
    echo "Running benchmark for model: $model"

    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $COT_WORKERS --barebone
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $WORKERS
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $WORKERS --rotate-shots --n-shots $N_SHOTS
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $WORKERS --ctx
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $WORKERS --rotate-shots --n-shots $N_SHOTS --ctx
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $COT_WORKERS --cot
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $COT_WORKERS --cot --rotate-shots --n-shots $N_SHOTS
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $COT_WORKERS --cot --ctx
    python src/run_benchmark.py --backend "$BACKEND" --prompt "$PROMPT" --benchmark "$BENCHMARK" --shots-benchmark "$SHOTS_BENCHMARK" --seed $SEED --model "$model" --workers $COT_WORKERS --cot --rotate-shots --n-shots $N_SHOTS --ctx

    echo "------------------------------------------------------"
done

echo "All benchmark runs completed successfully!"
