# TIME2025 LLM

[INTRODUCTION HERE]


## Setup and run the project
1. Prepare the working environment:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set the OpenRouter key:
    1. Create `.env-openrouter` file in the project root
    2. Write `OPENROUTER_API_KEY="your-key-here"` to the file, using your own API key.

3. Run the benchmarks:
```sh
./run_deepseekV3.sh
./run_gemma3.sh
./run_llama4.sh
./run_qwen3_235B.sh
./run_qwen3.sh
```

> [!IMPORTANT]  
> Qwen3 models have a specific feature that allows enabling/disabling the reasoning CoT generation by appending `/think` / `/no_think` to the prompt. For this reason all the non-CoT tests for the Qwen models are commented. You will need to manually add `/no_think` to the prompt for these tests to be correctly executed. Also set `max_tokens_no_cot` to 10.

4. Move the resuling JSONs to `results/data/`
5. Plot the overall results with the command `python3 results/data`
6. Plot all the results with the command `python src/generate_plots.py -d results/data`

> [!NOTE]  
> You may need to install LaTeX and add it to the PATH to correctly generate the plots.