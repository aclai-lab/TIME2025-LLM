from utils.benchmark import Benchmark, BenchmarkFormula, BenchmarkMetadata, FormulaType
from utils.logger import Logger


MAX_FORMULA_SIZE = 128


def parse_dth_line(line: str) -> tuple[int, str, str]:
    line_elements = line.strip().split(',')
    
    id_enriched = line_elements[0]
    size_premise = int(line_elements[1])
    size_conclusion = int(line_elements[2])
    size_total = int(line_elements[3])
    compositional_hops = int(line_elements[4])
    id_base = line_elements[5]
    id_plus = line_elements[6]
    base = line_elements[7]
    plus = line_elements[8]
    enriched = line_elements[9]
    enriched_valid = line_elements[10]
    enriched_invalid = line_elements[11]
    
    return size_total, enriched_valid, enriched_invalid

def remove_duplicates(formulas: list[BenchmarkFormula]) -> list[BenchmarkFormula]:
    seen_formulas = set()
    unique_formulas = []
    
    for formula in formulas:
        if formula not in seen_formulas:
            seen_formulas.add(formula)
            unique_formulas.append(formula)
            
    return unique_formulas

def convert_dth_file(input_path: str, output_path: str, remove_duplicates_flag: bool = False, filter: bool = False) -> None:
    formulas = []
    Logger.info('convert_dth_file', f'Reading {input_path} contents...')
    with open(input_path, 'r') as f:
        lines = f.readlines()

    Logger.info('convert_dth_file', f'Parsing formulas...')
    lines.pop(0)
    for line in lines:
        if not line.strip(): continue

        size_total, formula_valid_str, formula_invalid_str = parse_dth_line(line)
        
        if size_total > MAX_FORMULA_SIZE:
            continue
            
        if '|=' in formula_valid_str:
            premise, conclusion = formula_valid_str.split('|=')
            formula_valid_str = f'({premise.strip()}) -> ({conclusion.strip()})'
        formula_valid = BenchmarkFormula(formula=formula_valid_str, size=size_total, type=FormulaType.VALID)
        formulas.append(formula_valid)
        
        if '|=' in formula_invalid_str:
            premise, conclusion = formula_invalid_str.split('|=')
            formula_invalid_str = f'({premise.strip()}) -> ({conclusion.strip()})'
        formula_invalid = BenchmarkFormula(formula=formula_invalid_str, size=size_total, type=FormulaType.UNSATISFIABLE)
        formulas.append(formula_invalid)

    if remove_duplicates_flag:
        Logger.info('convert_dth_file', f'Removing duplicates...')
        original_count = len(formulas)
        formulas = remove_duplicates(formulas)
        Logger.info('convert_dth_file', f'Removed {original_count - len(formulas)} duplicate formulas')

    metadata = BenchmarkMetadata(
        num_instances=None,
        num_valid=None,
        num_satisfiable_not_valid=None,
        num_unsatisfiable=None,
        seed=None
    )
    
    Logger.info('convert_dth_file', f'Saving benchmark to {output_path}...')
    benchmark = Benchmark(metadata=metadata, formulas=formulas)
    benchmark.save(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DTH format file to Benchmark JSON')
    parser.add_argument('input_file', type=str, help='Input DTH file path')
    parser.add_argument('output_file', type=str, help='Output JSON file path')
    parser.add_argument('--remove-duplicates', action='store_true', help='Remove formulas with duplicate premises')
    parser.add_argument('--filter', action='store_true', help='Remove formulas containing 0, 1, xor, R, W, M')
    
    args = parser.parse_args()
    
    convert_dth_file(
        args.input_file,
        args.output_file,
        remove_duplicates_flag=args.remove_duplicates,
        filter=args.filter
    )

if __name__ == '__main__':
    main()
