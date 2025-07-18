import os
import json
import time
import argparse
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI as OpenRouter
from dotenv import load_dotenv

from utils.benchmark import Benchmark, BenchmarkFormula, FormulaType
from utils.logger import Logger
from utils.ansi import AnsiCodes as AC
from utils.prompt_builder import PromptBuilder, PromptOptions

MODEL_PARAMS = {
    "deepseek/deepseek-r1": {  # Source: https://huggingface.co/deepseek-ai/DeepSeek-R1#deepseek-r1-evaluation
        "max_tokens_cot": 30_000,
        "temperature_cot": 0.6,
        "top_p": 0.95,
        "extra_body": {
            "include_reasoning": True,
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False
            }
        }
    },
    "deepseek/deepseek-chat-v3-0324": {  # Source: https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/deepseek-v3-0324-how-to-run-locally
        "max_tokens_cot": 30_000,
        "temperature_cot": 0.3,
        "min_p": 0.0,
        "extra_body": {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False
            }
        }
    },
    "qwen/qwen3-32b": {  # Source: https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
        "max_tokens_cot": 30_000,
        "temperature_cot": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "extra_body": {
            "include_reasoning": True,
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False
            }
        }
    },
    "qwen/qwen3-235b-a22b": {  # Source: https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
        "max_tokens_cot": 30_000,
        "temperature_cot": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "extra_body": {
            "include_reasoning": True,
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False
            }
        }
    },
    "google/gemma-3-27b-it": {  # Source: https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune
        "max_tokens_cot": 16_000,
        "temperature_cot": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.0,
        "extra_body": {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False
            }
        }
    },
    "meta-llama/llama-4-maverick": {  # Source: https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune
        "max_tokens_cot": 16_000,
        "temperature_cot": 0.6,
        "top_p": 0.9,
        "min_p": 0.01,
        "extra_body": {
            "provider": {
                "order": ["Cent-ML"],
                "allow_fallbacks": False
            }
        }
    },
}

DEFAULT_PARAMS = {
    "max_tokens_cot": 8_000,
    "max_tokens_no_cot": 4,
    
    "temperature_cot": 0.6,
    "temperature_no_cot": 0.0,
    
    "top_p": None,
    "top_k": None,
    "min_p": None,
    
    "extra_body": None,
    "seed": None
}


CORRECT_DN = f'{AC.FG_GREEN}{AC.BOLD}✓{AC.RESET}'
WRONG_DN = f'{AC.FG_RED}{AC.BOLD}⨯{AC.RESET}'

@dataclass
class InferenceConfig:
    """Configuration parameters for LLM inference."""
    max_tokens: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    temp: float | None = None
    seed: int | None = None
    cot: bool | None = None
    n_shots: int | None = None
    include_context: bool | None = None
    rotate_shots: bool | None = None
    barebone: bool | None = None
    
    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

def load_env():
    load_dotenv('.env-openrouter')
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    return api_key


def get_model_params(model: str) -> dict:
    """Get model-specific parameters."""
    params = DEFAULT_PARAMS.copy()
    if model in MODEL_PARAMS:
        params.update(MODEL_PARAMS[model])
    else:
        Logger.warning('get_model_params', 'Model not recognized, using default params!')
    return params


def ask_llm_formula_validity(
    client: OpenRouter,
    model: str,
    formula: BenchmarkFormula,
    prompt_builder: PromptBuilder,
    config: InferenceConfig,
    extra_body: dict | None,
    max_attempts: int = 10,
    attempt_delay_ms: int = 5_000
) -> tuple[bool | None, str | None]:
    for attempt in range(max_attempts):
        try:
            # Build prompt with appropriate options
            task_option = PromptOptions.COT_TASK if config.cot else PromptOptions.NO_COT_TASK
            prompt_options = PromptOptions.BASE_TASK | task_option
            if config.include_context:
                prompt_options |= (PromptOptions.CONTEXT | PromptOptions.SYNTAX | PromptOptions.SEMANTICS)
            if config.barebone:
                prompt_options = PromptOptions.BAREBONE
            
            system_prompt = prompt_builder.build(
                prompt_options, 
                n_shots=config.n_shots, 
                cot=config.cot or config.barebone, 
                reset_seed=not config.rotate_shots
            )
            
            seed = None if config.seed == -1 else config.seed
            prompt = system_prompt + "\n\nFormula to evaluate: " + formula.formula
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_tokens,
                temperature=config.temp,
                extra_body=extra_body,
                seed=seed
            )

            if response.choices:
                original_result = response.choices[0].message.content
                
                reasoning = response.choices[0].message.reasoning
                if reasoning != None:
                    original_result = reasoning + "\n\n" + original_result
                
                result = original_result.strip().lower()  # type: ignore
                result = ''.join(char for char in result if char not in '.,:;\'\"()')
                result = result.strip()
                
                if not config.cot and not config.barebone:
                    if result == "valid" or result == "[valid]": return (True, original_result)
                    elif result == "invalid" or result == "[invalid]": return (False, original_result)
                    else:
                        Logger.warning('ask_llm_formula_validity', f'Unknown response: "{original_result}"')
                        #Logger.warning('ask_llm_formula_validity', f'Full response: {response.to_json()}')
                        return (None, result)
                else:
                    if "[valid]" in result: return (True, original_result)
                    elif "[invalid]" in result: return (False, original_result)
                    elif "\\boxed{valid}" in result: return (True, original_result)
                    elif "\\boxed{invalid}" in result: return (False, original_result)
                    elif "\\[valid\\]" in result: return (True, original_result)
                    elif "\\[invalid\\]" in result: return (False, original_result)
                    elif "\\[[valid\\]" in result: return (True, original_result)
                    elif "\\[[invalid\\]" in result: return (False, original_result)
                    elif "\\boxed{\\text{valid}}" in result: return (True, original_result)
                    elif "\\boxed{\\text{invalid}}" in result: return (False, original_result)
                    elif "\\boxed{[\\text{valid}]}" in result: return (True, original_result)
                    elif "\\boxed{[\\text{invalid}]}" in result: return (False, original_result)
                    else:
                        Logger.warning('ask_llm_formula_validity', f'Unknown response: {response.to_json()}')
                        return (None, original_result)
            else:
                Logger.error('ask_llm_formula_validity', f'Response text was empty. Full response: {response.to_json()}', faint=True)
                if attempt+1 >= max_attempts:
                    raise Exception(f'unable to get a response after {max_attempts} attempts')
                else:
                    Logger.warning('ask_llm_formula_validity', f'attempt {attempt+1}/{max_attempts} failed, retrying after {attempt_delay_ms}ms...')
                    time.sleep(attempt_delay_ms / 1000.0)
        
        except Exception as e:
            Logger.error("ask_llm_formula_validity", f"Attempt {attempt+1}/{max_attempts} failed with error: {str(e)}")
            if attempt+1 >= max_attempts:
                return (None, None)
            else:
                Logger.warning('ask_llm_formula_validity', f'Retrying after {attempt_delay_ms}ms...')
                time.sleep(attempt_delay_ms / 1000.0)

    return (None, None)
        

def run_benchmark(
    client: OpenRouter,
    model: str, 
    benchmark: Benchmark,
    prompt_builder: PromptBuilder,
    config: InferenceConfig,
    extra_body: dict | None,
    workers: int = 10
) -> dict:
    total = benchmark.metadata.num_instances
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    raw_results = []  # Store raw results for each formula

    def process_formula(formula: BenchmarkFormula):
        return (formula, ask_llm_formula_validity(client, model, formula, prompt_builder, config, extra_body))
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_formula, formula) for formula in benchmark.formulas]
        
        for future in as_completed(futures):
            formula, response_data = future.result()
            llm_said_valid = response_data[0]
            response = response_data[1]
            
            if llm_said_valid is None:
                Logger.warning('run_benchmark', 'ignoring current instance due to missing answer')
                continue

            formula_is_valid = formula.type == FormulaType.VALID
            correct = (llm_said_valid and formula_is_valid) or (not llm_said_valid and not formula_is_valid)
            
            # Store raw result
            raw_result = {
                "formula": {
                    "formula": formula.formula,
                    "size": formula.size,
                    "type": formula.type.name
                },
                "response": response,
                "response_is_correct": correct
            }
            raw_results.append(raw_result)

            if llm_said_valid and formula_is_valid:             true_positives += 1;  Logger.info("run_benchmark", f"✓ (TP) {str(formula)}")
            elif not llm_said_valid and not formula_is_valid:   true_negatives += 1;  Logger.info("run_benchmark", f"✓ (TN) {str(formula)}")
            elif llm_said_valid and not formula_is_valid:       false_positives += 1; Logger.info("run_benchmark", f"⨯ (FP) {str(formula)}")
            else:                                               false_negatives += 1; Logger.info("run_benchmark", f"⨯ (FN) {str(formula)}")

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    Logger.info('run_benchmark', f'Model: {model}', faint=False)
    Logger.info('run_benchmark', f'Confusion Matrix:', faint=False)
    Logger.info('run_benchmark', f'\tTrue Positives:  {CORRECT_DN} {true_positives}', faint=False)
    Logger.info('run_benchmark', f'\tTrue Negatives:  {CORRECT_DN} {true_negatives}', faint=False)
    Logger.info('run_benchmark', f'\tFalse Positives: {WRONG_DN} {false_positives}', faint=False)
    Logger.info('run_benchmark', f'\tFalse Negatives: {WRONG_DN} {false_negatives}', faint=False)
    Logger.info('run_benchmark', f'Metrics:', faint=False)
    Logger.info('run_benchmark', f'\tAccuracy:  {round(accuracy*100, 2)}%', faint=False)
    Logger.info('run_benchmark', f'\tPrecision: {round(precision*100, 2)}%', faint=False)
    Logger.info('run_benchmark', f'\tRecall:    {round(recall*100, 2)}%', faint=False)
    Logger.info('run_benchmark', f'\tF1 Score:  {round(f1_score*100, 2)}%', faint=False)

    return {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "confusion_matrix": {
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        },
        "raw_results": raw_results
    }


def main():
    parser = argparse.ArgumentParser(description='Run LTL benchmark')
    parser.add_argument('--backend', type=str, choices=['ollama', 'openrouter'], required=True, help='API backend to use)')
    parser.add_argument('--model', type=str, help='Model to use', required=True)
    parser.add_argument('--benchmark', type=str, required=True, help='Path to benchmark JSON file for testing')
    parser.add_argument('--shots-benchmark', type=str, required=True, help='Path to benchmark JSON file for prompt shots')
    parser.add_argument('--seed', type=int, default=-1, help='Seed for API randomness (default is random)')
    parser.add_argument('--temp', type=float, help='Temperature for API sampling (default is 0)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers for API requests')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--prompt', type=str, default='resources/prompt.json', help='Path to prompt template JSON file')
    parser.add_argument('--n-shots', type=int, default=0, help='Number of examples to include in the prompt, omitted if set to zero (note that if n-shots=5, there will be 10 shots; 5 valid and 5 invalid)')
    parser.add_argument('--cot', action='store_true', help='Enable Chain of Thought reasoning')
    parser.add_argument('--ctx', action='store_true', help='Include context, syntax and semantics in prompt')
    parser.add_argument('--barebone', action='store_true', help='Use only a minimal instruction')
    parser.add_argument('--rotate-shots', action='store_true', help='Use new shots for each evaluated formula')
    args = parser.parse_args()

    # If the output is not specified, use model's name
    if not args.output:
        args.output = f"{args.model.split('/')[-1]}>"
        if args.barebone: args.output += f"-barebone"
        if args.n_shots: args.output += f"-{args.n_shots * 2}shot"
        if args.ctx: args.output += f"-ctx"
        if args.cot: args.output += f"-cot"
        args.output += '.json'
    
    try:
        base_url = 'https://openrouter.ai/api/v1' if args.backend == 'openrouter' else 'http://localhost:11434/v1'

        # Load Openrouter credentials
        api_key = load_env()
        client = OpenRouter(
            base_url=base_url,
            api_key=api_key,
            timeout=900
        )
        
        # Load benchmarks
        benchmark = Benchmark.load(args.benchmark)
        shots_benchmark = Benchmark.load(args.shots_benchmark)
        
        Logger.info("main", f"Running benchmark {args.benchmark} using {args.model}")
        Logger.info("main", f"Using benchmark {args.shots_benchmark} for prompt shots")
        
        # Create prompt builder
        prompt_builder = PromptBuilder(
            prompt_json_path=args.prompt,
            benchmark=shots_benchmark,
            seed=args.seed,
        )
        
        # Get model parameters
        model_params = get_model_params(model=args.model)
        
        # Create inference configuration with model-specific parameters
        inference_config = InferenceConfig(
            # Set temperature based on chain of thought
            temp=args.temp if args.temp != None else (
                model_params['temperature_cot'] if args.cot or args.barebone else model_params['temperature_no_cot']
            ),
            # Set max tokens based on chain of thought
            max_tokens=model_params['max_tokens_cot'] if args.cot or args.barebone else model_params['max_tokens_no_cot'],
            # Set other model parameters
            top_p=model_params['top_p'],
            top_k=model_params['top_k'],
            min_p=model_params['min_p'],
            # Set configuration parameters
            seed=args.seed,
            cot=args.cot,
            n_shots=args.n_shots,
            include_context=args.ctx,
            rotate_shots=args.rotate_shots,
            barebone=args.barebone
        )
        
        # Prepare extra_body with additional parameters if needed
        extra_body = model_params['extra_body'].copy() if model_params['extra_body'] else {}
        if inference_config.top_p != None:
            extra_body["top_p"] = inference_config.top_p
        if inference_config.top_k != None:
            extra_body["top_k"] = inference_config.top_k
        if inference_config.min_p != None:
            extra_body["min_p"] = inference_config.min_p
            
        Logger.info('main', f'Inference config: {inference_config.to_dict()}')
        Logger.info('main', f'Extra body: {extra_body}')
        
        # Run benchmark
        results = run_benchmark(
            client=client,
            model=args.model,
            benchmark=benchmark,
            prompt_builder=prompt_builder,
            config=inference_config,
            extra_body=extra_body,
            workers=args.workers
        )
        
        # Save results if output specified
        if args.output:
            output_results = {
                "model": args.model,
                "results": {**results},
                "benchmark_metadata": {**asdict(benchmark.metadata)},
                "parameters": {**asdict(inference_config)}
            }
            with open(args.output, 'w') as f:
                json.dump(output_results, f, indent=2)
            Logger.info("main", f"Results saved to {args.output}")
            
    except Exception as e:
        Logger.error("main", str(e))
        exit(1)

if __name__ == "__main__":
    main()
