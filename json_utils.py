import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

def parse_filename(filename: str) -> Dict[str, Any]:
    # Extract model name (everything before '>')
    filename = os.path.basename(filename)
    if filename.endswith('.json'):
        filename = filename[:-5]  # Remove .json extension
    
    model_match = re.match(r'([^>]+)>', filename)
    if model_match:
        model_name = model_match.group(1)
    else:
        model_name = filename
    
    # Check for configuration flags
    has_6shot = '6shot' in filename
    has_cot = 'cot' in filename
    has_ctx = 'ctx' in filename
    has_barebone = 'barebone' in filename
    
    return {
        'model_name': model_name,
        'n_shots': 6 if has_6shot else 0,
        'cot': has_cot,
        'include_context': has_ctx,
        'barebone': has_barebone
    }

def format_config_name(config: str) -> str:
    if not config or config == "base":
        return "Base"
    
    parts = config.split('-')
    formatted_parts = []
    
    for part in parts:
        if part == "cot":
            formatted_parts.append("CoT")
        elif part == "ctx":
            formatted_parts.append("Context")
        elif part.endswith("shot"):
            # Extract number from "6shot" format
            num = part.replace("shot", "")
            formatted_parts.append(f"{num}-Shot")
        elif part == "barebone":
            formatted_parts.append("Barebone")
        else:
            # Capitalize first letter of other parts
            formatted_parts.append(part.capitalize())
    
    return " ".join(formatted_parts)

def create_config_str(n_shots: int, include_context: bool, cot: bool, barebone: bool) -> str:
    config_parts = []
    if n_shots > 0:
        config_parts.append(f"{n_shots}shot")
    if include_context:
        config_parts.append("ctx")
    if cot:
        config_parts.append("cot")
    if barebone:
        config_parts.append("barebone")
    
    return "-".join(config_parts) if config_parts else "base"

def calculate_accuracies(raw_results: List[Dict]) -> Tuple[float, float, float]:
    valid_correct = 0
    valid_total = 0
    invalid_correct = 0
    invalid_total = 0
    
    for result in raw_results:
        formula = result.get('formula', {})
        formula_type = formula.get('type', '')
        is_correct = result.get('response_is_correct', False)
        
        if formula_type == 'VALID':
            valid_total += 1
            if is_correct:
                valid_correct += 1
        else:  # Assume anything not VALID is INVALID/UNSATISFIABLE
            invalid_total += 1
            if is_correct:
                invalid_correct += 1
    
    valid_accuracy = valid_correct / valid_total if valid_total > 0 else 0.0
    invalid_accuracy = invalid_correct / invalid_total if invalid_total > 0 else 0.0
    overall_accuracy = (valid_correct + invalid_correct) / (valid_total + invalid_total) if (valid_total + invalid_total) > 0 else 0.0
    
    return valid_accuracy, invalid_accuracy, overall_accuracy

def process_file(json_file: Path) -> Optional[Dict]:
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract model from the JSON if available, otherwise from filename
        model_name = data.get('model', '')
        if not model_name:
            filename_info = parse_filename(json_file.stem)
            model_name = filename_info['model_name']
        
        # Extract parameters and raw_results
        parameters = data.get('parameters', {})
        raw_results = data.get('results', {}).get('raw_results', [])
        
        if not raw_results:
            print(f"No raw_results found in {json_file}")
            return None
        
        # Get configuration from parameters or filename
        filename_info = parse_filename(json_file.stem)
        
        cot = parameters.get('cot', filename_info['cot'])
        n_shots = parameters.get('n_shots', filename_info['n_shots'])
        include_context = parameters.get('include_context', filename_info['include_context'])
        barebone = parameters.get('barebone', filename_info['barebone'])
        
        # Create a configuration string
        config_str = create_config_str(n_shots, include_context, cot, barebone)
        formatted_config = format_config_name(config_str)
        
        # Calculate accuracy metrics
        valid_accuracy, invalid_accuracy, overall_accuracy = calculate_accuracies(raw_results)
        
        # Create a short model name for display
        short_model = model_name.split('/')[-1] if '/' in model_name else model_name
        
        return {
            'model': short_model,
            'config': config_str,
            'formatted_config': formatted_config,
            'valid_accuracy': valid_accuracy,
            'invalid_accuracy': invalid_accuracy,
            'overall_accuracy': overall_accuracy,
            'cot': cot,
            'n_shots': n_shots,
            'include_context': include_context,
            'barebone': barebone,
            'filename': json_file.stem
        }
    
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        return None

def process_directory(directory: str) -> List[Dict]:
    # Get all JSON files in the directory
    json_files = list(Path(directory).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return []
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    # Store results for each configuration
    results = []
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        model_config = process_file(json_file)
        if model_config:
            results.append(model_config)
    
    return results
