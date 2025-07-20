import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.fragment_finder import analyze_formula_complexity, Complexity

def load_json_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{filepath}': {e}")
        sys.exit(1)

def analyze_formulas(formulas, verbose=False):
    complexity_groups = defaultdict(list)
    
    print(f"Analyzing {len(formulas)} formulas...")
    
    for i, formula_data in enumerate(formulas):
        if verbose and (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(formulas)} formulas")
            
        formula = formula_data['formula']
        try:
            complexity = analyze_formula_complexity(formula, verbose=False)
            complexity_groups[complexity].append(formula_data)
        except Exception as e:
            if verbose:
                print(f"Error analyzing formula '{formula}': {e}")
            continue
    
    return complexity_groups

def sample_balanced_dataset(complexity_groups, samples_per_class=100):
    balanced_dataset = []
    complexity_counts = {}
    
    print(f"\nCreating balanced dataset with {samples_per_class} samples per complexity class...")
    print("Ensuring equal valid/invalid distribution within each complexity class...")
    
    for complexity, formulas in complexity_groups.items():
        # Separate formulas by type (valid vs invalid)
        valid_formulas = [f for f in formulas if f['type'] == 'valid']
        invalid_formulas = [f for f in formulas if f['type'] in ['unsatisfiable', 'satisfiable_not_valid']]
        
        # Calculate how many of each type we need
        samples_per_type = samples_per_class // 2
        remaining_samples = samples_per_class % 2
        
        available_valid = len(valid_formulas)
        available_invalid = len(invalid_formulas)
        
        # Determine actual samples we can get
        actual_valid = min(samples_per_type, available_valid)
        actual_invalid = min(samples_per_type, available_invalid)
        
        # If we have remaining sample and more of one type available, use it
        if remaining_samples > 0:
            if available_valid > actual_valid and available_invalid <= actual_invalid:
                actual_valid += remaining_samples
            elif available_invalid > actual_invalid and available_valid <= actual_valid:
                actual_invalid += remaining_samples
            else:
                # Distribute remaining sample to the type with more available samples
                if available_valid - actual_valid >= available_invalid - actual_invalid:
                    actual_valid += remaining_samples
                else:
                    actual_invalid += remaining_samples
        
        sampled_formulas = []
        
        # Sample valid formulas
        if actual_valid > 0:
            sampled_valid = random.sample(valid_formulas, actual_valid)
            sampled_formulas.extend(sampled_valid)
        
        # Sample invalid formulas
        if actual_invalid > 0:
            sampled_invalid = random.sample(invalid_formulas, actual_invalid)
            sampled_formulas.extend(sampled_invalid)
        
        total_sampled = len(sampled_formulas)
        balanced_dataset.extend(sampled_formulas)
        
        complexity_counts[complexity] = {
            'total': total_sampled,
            'valid': actual_valid,
            'invalid': actual_invalid,
            'available_valid': available_valid,
            'available_invalid': available_invalid
        }
        
        print(f"{complexity.name}: {total_sampled}/{samples_per_class} samples "
              f"(valid: {actual_valid}/{available_valid}, invalid: {actual_invalid}/{available_invalid})")
    
    return balanced_dataset, complexity_counts

def create_output_json(balanced_dataset, original_metadata, complexity_counts):
    # Count types in balanced dataset
    type_counts = Counter(formula['type'] for formula in balanced_dataset)
    
    # Create detailed complexity breakdown
    complexity_breakdown = {}
    for complexity, counts in complexity_counts.items():
        if isinstance(counts, dict):
            complexity_breakdown[complexity.name] = {
                'total': counts['total'],
                'valid': counts['valid'],
                'invalid': counts['invalid']
            }
        else:
            # Fallback for old format
            complexity_breakdown[complexity.name] = {'total': counts}
    
    output_data = {
        "metadata": {
            "num_instances": len(balanced_dataset),
            "num_valid": type_counts.get('valid', 0),
            "num_satisfiable_not_valid": type_counts.get('satisfiable_not_valid', 0),
            "num_unsatisfiable": type_counts.get('unsatisfiable', 0),
            "seed": original_metadata.get('seed'),
            "sampling_method": "complexity_balanced_valid_invalid",
            "samples_per_complexity": 100,
            "complexity_breakdown": complexity_breakdown
        },
        "formulas": balanced_dataset
    }
    
    return output_data

def create_visualizations(balanced_dataset, output_dir="plots"):
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze complexities and sizes
    formula_analysis = []
    for formula_data in balanced_dataset:
        complexity = analyze_formula_complexity(formula_data['formula'])
        formula_analysis.append({
            'complexity': complexity.name,
            'size': formula_data['size'],
            'formula': formula_data['formula']
        })
    
    # Create complexity bar plot
    plt.figure(figsize=(12, 6))
    complexity_counts = Counter(item['complexity'] for item in formula_analysis)
    
    plt.subplot(1, 2, 1)
    complexities = list(complexity_counts.keys())
    counts = list(complexity_counts.values())
    
    bars = plt.bar(complexities, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Distribution of Formula Complexities', fontsize=14, fontweight='bold')
    plt.xlabel('Complexity Class', fontsize=12)
    plt.ylabel('Number of Formulas', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create KDE plot for formula sizes by complexity
    plt.subplot(1, 2, 2)
    
    # Group sizes by complexity
    size_by_complexity = defaultdict(list)
    for item in formula_analysis:
        size_by_complexity[item['complexity']].append(item['size'])
    
    # Plot KDE for each complexity class
    colors = plt.cm.Set3(np.linspace(0, 1, len(size_by_complexity)))
    
    for i, (complexity, sizes) in enumerate(size_by_complexity.items()):
        if len(sizes) > 1:  # Need at least 2 points for KDE
            try:
                sns.kdeplot(data=sizes, label=complexity, color=colors[i], alpha=0.7)
            except:
                # Fallback to histogram if KDE fails
                plt.hist(sizes, alpha=0.5, label=complexity, density=True, 
                        bins=min(10, len(set(sizes))))
        else:
            # For single values, plot a vertical line
            plt.axvline(x=sizes[0], color=colors[i], linestyle='--', 
                       alpha=0.7, label=f"{complexity} (size={sizes[0]})")
    
    plt.title('Formula Size Distribution by Complexity', fontsize=14, fontweight='bold')
    plt.xlabel('Formula Size', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/complexity_analysis.png", dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir}/complexity_analysis.png")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("="*60)
    
    for complexity in sorted(size_by_complexity.keys()):
        sizes = size_by_complexity[complexity]
        if sizes:
            # Get valid/invalid breakdown for this complexity
            complexity_formulas = [item for item in formula_analysis if item['complexity'] == complexity]
            valid_sizes = [item['size'] for item in complexity_formulas 
                          if any(f['formula'] == item['formula'] and f['type'] == 'valid' 
                                for f in balanced_dataset)]
            invalid_sizes = [item['size'] for item in complexity_formulas 
                            if any(f['formula'] == item['formula'] and f['type'] in ['unsatisfiable', 'satisfiable_not_valid'] 
                                  for f in balanced_dataset)]
            
            print(f"\n{complexity}:")
            print(f"  Total count: {len(sizes)}")
            print(f"  Valid: {len(valid_sizes)}, Invalid: {len(invalid_sizes)}")
            print(f"  Size range: {min(sizes)} - {max(sizes)}")
            print(f"  Mean size: {np.mean(sizes):.2f}")
            print(f"  Std dev: {np.std(sizes):.2f}")
            if valid_sizes:
                print(f"  Valid mean size: {np.mean(valid_sizes):.2f}")
            if invalid_sizes:
                print(f"  Invalid mean size: {np.mean(invalid_sizes):.2f}")
    
    # Print overall balance summary
    print(f"\n" + "="*60)
    print("BALANCE SUMMARY")
    print("="*60)
    total_valid = sum(1 for f in balanced_dataset if f['type'] == 'valid')
    total_invalid = len(balanced_dataset) - total_valid
    print(f"Overall dataset: {len(balanced_dataset)} formulas")
    print(f"  Valid: {total_valid} ({total_valid/len(balanced_dataset)*100:.1f}%)")
    print(f"  Invalid: {total_invalid} ({total_invalid/len(balanced_dataset)*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Analyze formula complexity and create balanced dataset")
    
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('-o', '--output', default='balanced_formulas.json', help='Output JSON file path (default: balanced_formulas.json)')
    parser.add_argument('-s', '--samples', type=int, default=100, help='Number of samples per complexity class (default: 100)')
    parser.add_argument('--plot-dir', default='plots', help='Directory to save plots (default: plots)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible sampling')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Load input data
    print(f"Loading data from {args.input_file}...")
    data = load_json_data(args.input_file)
    
    if 'formulas' not in data:
        print("Error: JSON file must contain 'formulas' key.")
        sys.exit(1)
    
    formulas = data['formulas']
    print(f"Loaded {len(formulas)} formulas")
    
    # Analyze formulas by complexity
    complexity_groups = analyze_formulas(formulas, args.verbose)
    
    # Print complexity distribution
    print(f"\nComplexity distribution in input data:")
    for complexity in Complexity:
        formulas_in_class = complexity_groups[complexity]
        valid_count = len([f for f in formulas_in_class if f['type'] == 'valid'])
        invalid_count = len([f for f in formulas_in_class if f['type'] in ['unsatisfiable', 'satisfiable_not_valid']])
        total_count = len(formulas_in_class)
        print(f"  {complexity.name}: {total_count} formulas (valid: {valid_count}, invalid: {invalid_count})")
    
    # Create balanced dataset
    balanced_dataset, complexity_counts = sample_balanced_dataset(
        complexity_groups, args.samples
    )
    
    # Create output JSON
    output_data = create_output_json(balanced_dataset, data.get('metadata', {}), complexity_counts)
    
    # Save output file
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBalanced dataset saved to {args.output}")
    print(f"Total formulas in output: {len(balanced_dataset)}")
    
    # Create visualizations
    create_visualizations(balanced_dataset, args.plot_dir)

if __name__ == '__main__':
    main()
