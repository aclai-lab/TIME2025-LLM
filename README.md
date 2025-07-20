# TIME2025 LLM

This repository contains the code required to generate and evaluate the benchmarks shown in the *Assessing The (In)Ability of LLMs To Reason in Interval Temporal Logic* paper submitted to the 2025 [TIME](https://time2025conf.github.io/) conference.


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

4. Move the resulting JSONs to `results/data/`
5. Plot the overall results with the command `python3 plotter.py results/data`
6. Plot all the results with the command `python3 src/generate_plots.py -d results/data`
7. To test the complexity class accuracy:
    1. Run a benchmark evaluation using `resources/benchmark/complexity_100.json` or generate your own by running `python3 src/benchmark_to_complexity.py resources/benchmark/all.json`
    2. Plot the result with `python3 src/complexity_plot.py 'result-path-here'`

> [!NOTE]  
> For the complexity class tests you can use any benchmark JSON, but they will not be balanced.

> [!NOTE]  
> You may need to install LaTeX and add it to the PATH to correctly generate the plots.