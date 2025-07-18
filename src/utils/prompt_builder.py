from enum import Flag, auto
import json
import random
from utils.logger import Logger

from utils.benchmark import Benchmark, BenchmarkFormula, FormulaType


class PromptOptions(Flag):
    BAREBONE = auto()
    CONTEXT = auto()
    SYNTAX = auto()
    SEMANTICS = auto()
    BASE_TASK = auto()
    COT_TASK = auto()
    NO_COT_TASK = auto()


class PromptBuilder:
    def __init__(self, prompt_json_path: str, benchmark: Benchmark, seed: int) -> None:
        try:
            with open(prompt_json_path, 'r', encoding='utf-8') as file:
                self.prompt_templates: dict[str, str] = json.load(file)
                Logger.info(source="PromptBuilder.init", message=f"Successfully loaded prompt templates from {prompt_json_path}")
        except FileNotFoundError:
            Logger.error(source="PromptBuilder.init", message=f"Prompt template file not found: {prompt_json_path}")
            raise
        except json.JSONDecodeError as e:
            Logger.error(source="PromptBuilder.init", message=f"Invalid JSON format in {prompt_json_path}: {str(e)}")
            raise
        except Exception as e:
            Logger.error(source="PromptBuilder.init", message=f"Unexpected error while loading {prompt_json_path}: {str(e)}")
            raise

        self.benchmark = benchmark
        self.seed = seed
        random.seed(seed)

        self.valid_formulas: list[BenchmarkFormula] = [formula for formula in self.benchmark.formulas if formula.type == FormulaType.VALID]
        self.not_valid_formulas: list[BenchmarkFormula] = [formula for formula in self.benchmark.formulas if formula.type == FormulaType.UNSATISFIABLE or formula.type == FormulaType.SATISFIABLE_NOT_VALID]
    
    
    def build(self, options: PromptOptions, n_shots: int, cot: bool, reset_seed: bool = False) -> str:
        prompt_chunks: list[str] = []

        if PromptOptions.BAREBONE in options:
            return self.prompt_templates[PromptOptions.BAREBONE.name] # type: ignore

        if PromptOptions.BASE_TASK not in options:
            raise ValueError(f'{PromptOptions.BASE_TASK.name} must be included')

        if PromptOptions.COT_TASK not in options and PromptOptions.NO_COT_TASK not in options:
            raise ValueError(f'At least one option betwen {PromptOptions.COT_TASK.name} and {PromptOptions.NO_COT_TASK.name} must be included')
        
        if PromptOptions.COT_TASK in options and PromptOptions.NO_COT_TASK in options:
            raise ValueError(f'You cannot include both {PromptOptions.COT_TASK.name} and {PromptOptions.NO_COT_TASK.name}')

        for option in PromptOptions:
            if option in options:
                prompt_chunks.append(self.prompt_templates[option.name]) # type: ignore

        prompt_chunks = [c for c in prompt_chunks if len(c.strip()) > 0]
        
        if reset_seed:
            random.seed(self.seed)
        
        if n_shots != 0:
            if n_shots > 0:
                valid_examples = random.sample(self.valid_formulas, n_shots)
                not_valid_examples = random.sample(self.not_valid_formulas, n_shots)

                valid_tag = "[VALID]" if cot else "VALID"
                invalid_tag = "[INVALID]" if cot else "INVALID"

                valid_examples =     [f'{formula.formula}: {valid_tag}' for formula in valid_examples]
                not_valid_examples = [f'{formula.formula}: {invalid_tag}' for formula in not_valid_examples]
                all_examples = valid_examples + not_valid_examples
                random.shuffle(all_examples)
                examples_chunk = '# **Examples of Valid and Invalid instances**\n' + '\n\n'.join(all_examples)

                prompt_chunks.append(examples_chunk)
            else:
                Logger.error('prompt_builder.build', 'n_shot parameter is non-zero but negative')
                raise ValueError('n_shot parameter is non-zero but negative')

        prompt = '\n\n'.join(prompt_chunks)

        return prompt
