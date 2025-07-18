import re
from enum import Enum

from utils.logger import Logger

class Complexity(Enum):
    UNDECIDABLE = 1
    NON_PRIMITIVE_RECURSIVE = 2
    EXP_SPACE_COMPLETE = 3
    NEXPTIME_COMPLETE = 4
    NP_COMPLETE = 5
    
OP_TO_CHAR = {
    'meets': 'a',
    'later': 'l',
    'begins': 'b',
    'finished': 'e',
    'during': 'd',
    'overlaps': 'o',
    'met_by': 'A',
    'before': 'L',
    'begun_by': 'B',
    'ended_by': 'E',
    'contains': 'D',
    'overlapped_by': 'O'
}

# If a key is subset of the set, remove the value
# For example: if {'B', 'E', 'D'} is a subset of S, remove 'D' from S
MINIMAL_CONVERSION = {
    frozenset({'a', 'l'}): 'l',
    frozenset({'A', 'L'}): 'L',
    frozenset({'b', 'e', 'd'}): 'd',
    frozenset({'B', 'E', 'D'}): 'D',
    frozenset({'b', 'E', 'O'}): 'O',
    frozenset({'B', 'e', 'o'}): 'o',
}

FRAGMENT_COMPLEXITY = {
    frozenset({'a', 'A', 'b', 'B'}): Complexity.UNDECIDABLE,
    frozenset({'A', 'b', 'B', 'l'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'A', 'b'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'A', 'B'}): Complexity.UNDECIDABLE,
    frozenset({'A', 'b', 'l'}): Complexity.UNDECIDABLE,
    frozenset({'A', 'B', 'l'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'A', 'e', 'E'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'A', 'e'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'A', 'E'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'e', 'E', 'L'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'e'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'E'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'e', 'E'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'e', 'L'}): Complexity.UNDECIDABLE,
    frozenset({'a', 'E', 'L'}): Complexity.UNDECIDABLE,
    frozenset({'A', 'b'}): Complexity.NON_PRIMITIVE_RECURSIVE,
    frozenset({'A', 'B'}): Complexity.NON_PRIMITIVE_RECURSIVE,
    frozenset({'A', 'b', 'B'}): Complexity.NON_PRIMITIVE_RECURSIVE,
    frozenset({'a', 'b', 'B', 'L'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'A', 'e', 'E', 'l'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'a', 'b'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'a', 'B'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'a', 'b', 'B'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'a', 'b', 'L'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'a', 'B', 'L'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'A', 'e'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'A', 'E'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'A', 'e', 'E'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'A', 'e', 'l'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'A', 'E', 'l'}): Complexity.EXP_SPACE_COMPLETE,
    frozenset({'a', 'A'}): Complexity.NEXPTIME_COMPLETE,
    frozenset({'a'}): Complexity.NEXPTIME_COMPLETE,
    frozenset({'A'}): Complexity.NEXPTIME_COMPLETE,
    frozenset({'a', 'L'}): Complexity.NEXPTIME_COMPLETE,
    frozenset({'A', 'l'}): Complexity.NEXPTIME_COMPLETE,
    frozenset({'b'}): Complexity.NP_COMPLETE,
    frozenset({'B'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'B'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'B', 'l'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'B', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'B', 'l', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'l'}): Complexity.NP_COMPLETE,
    frozenset({'B', 'l'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'B', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'b', 'l', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'B', 'l', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'e'}): Complexity.NP_COMPLETE,
    frozenset({'E'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'E'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'E', 'l'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'E', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'E', 'l', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'E', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'l'}): Complexity.NP_COMPLETE,
    frozenset({'E', 'l'}): Complexity.NP_COMPLETE,
    frozenset({'e', 'l', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'E', 'l', 'L'}): Complexity.NP_COMPLETE,
    frozenset({'l'}): Complexity.NP_COMPLETE,
    frozenset({'L'}): Complexity.NP_COMPLETE,
    frozenset({'l', 'L'}): Complexity.NP_COMPLETE
}

def extract_operators(formula: str) -> set[str]:
    # Find all <operator> or [operator] patterns
    operators = re.findall(r'[<\[](\w+)[>\]]', formula)
    # Convert to set to remove duplicates
    return set(operators)

def convert_to_chars(operators: set[str]) -> set[str]:
    chars = set()
    for op in operators:
        if op in OP_TO_CHAR:
            chars.add(OP_TO_CHAR[op])
    return chars

def minimize_operators(chars: set[str]) -> set[str]:
    chars = set(chars)  # Create a copy to modify
    conversion_keys = sorted(MINIMAL_CONVERSION.keys(), key=lambda x: -len(x))
    
    for key in conversion_keys:
        if key.issubset(chars):
            chars.remove(MINIMAL_CONVERSION[key])
    return chars

def determine_complexity(chars: set[str]):
    # Sort complexity classes from highest to lowest (NP_COMPLETE first)
    complexity_order = sorted(Complexity, key=lambda x: -x.value)
    
    # Check if any of the fragments in FRAGMENT_COMPLEXITY is a subset of chars
    for complexity in complexity_order:
        for fragment in FRAGMENT_COMPLEXITY:
            if FRAGMENT_COMPLEXITY[fragment] == complexity and fragment == chars:
                return complexity
    
    return Complexity.UNDECIDABLE

def analyze_formula_complexity(formula, verbose=False):
    # Step 1: Extract operators
    operators = extract_operators(formula)
    Logger.info('analyze_formula_complexity', f'Extracted operators: {operators}', verbose=verbose)

    # Step 2: Convert to chars
    chars = convert_to_chars(operators)
    Logger.info('analyze_formula_complexity', f'Converted to chars: {chars}', verbose=verbose)
    
    # Step 3: Minimize the set
    minimal_chars = minimize_operators(chars)
    Logger.info('analyze_formula_complexity', f'Minimized chars: {minimal_chars}', verbose=verbose)
    
    # Step 4: Determine complexity
    complexity = determine_complexity(minimal_chars)
    return complexity

if __name__ == '__main__':
    formula = "(!<met_by>[begins]!(!r -> !((r & <begins><begun_by>p) | (r & <begins>q))))"
    complexity = analyze_formula_complexity(formula)
    print(f"\nThe formula's complexity is: {complexity}")
