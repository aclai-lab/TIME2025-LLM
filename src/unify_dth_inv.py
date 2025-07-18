import json
import argparse

from utils.benchmark import FormulaType

def main():
    parser = argparse.ArgumentParser(description='Create balanced formula datasets.')
    parser.add_argument('full_file', help='JSON file containing all formulas')
    parser.add_argument('output_file', help='Output JSON file')
    parser.add_argument('--n-instances', type=int, default=1000, help='Total number of instances in output file (default: 1000)')
    args = parser.parse_args()
    
    # Load JSON files
    with open(args.full_file, 'r') as f:
        all_data = json.load(f)

    valid_data = [formula for formula in all_data['formulas'] if formula['type'] == FormulaType.VALID.value]
    unsat_data = [formula for formula in all_data['formulas'] if formula['type'] != FormulaType.VALID.value]
    
    # Calculate required formulas
    main_instances_per_type = args.n_instances // 2
    shots_instances_per_type = args.n_instances * 2
    total_valid_needed = main_instances_per_type + shots_instances_per_type
    total_unsat_needed = main_instances_per_type + shots_instances_per_type
    
    # Randomly select formulas without overlap
    selected_valid = valid_data[:total_valid_needed]
    selected_unsat = unsat_data[:total_unsat_needed]
    
    # Split into main and shots datasets
    main_valid = selected_valid[:main_instances_per_type]
    shots_valid = selected_valid[main_instances_per_type:]
    
    main_unsat = selected_unsat[:main_instances_per_type]
    shots_unsat = selected_unsat[main_instances_per_type:]
    
    # Create main output dataset
    main_formulas = main_valid + main_unsat
    
    main_output_data = {
        "metadata": {
            "num_instances": args.n_instances,
            "num_valid": main_instances_per_type,
            "num_satisfiable_not_valid": 0,
            "num_unsatisfiable": main_instances_per_type,
            "seed": None
        },
        "formulas": main_formulas
    }
    
    # Create shots output dataset
    shots_formulas = shots_valid + shots_unsat
    
    shots_output_data = {
        "metadata": {
            "num_instances": shots_instances_per_type * 2,
            "num_valid": shots_instances_per_type,
            "num_satisfiable_not_valid": 0,
            "num_unsatisfiable": shots_instances_per_type,
            "seed": None
        },
        "formulas": shots_formulas
    }
    
    # Write output files
    with open(args.output_file, 'w') as f:
        json.dump(main_output_data, f, indent=2)
    
    shots_file = f"{args.output_file.rsplit('.', 1)[0]}_shots.{args.output_file.rsplit('.', 1)[1]}"
    with open(shots_file, 'w') as f:
        json.dump(shots_output_data, f, indent=2)
    
    print(f"Created main dataset with {args.n_instances} formulas")
    print(f"- {main_instances_per_type} valid")
    print(f"- {main_instances_per_type} unsatisfiable")
    
    print(f"\nCreated shots dataset with {shots_instances_per_type * 2} formulas")
    print(f"- {shots_instances_per_type} valid")
    print(f"- {shots_instances_per_type} unsatisfiable")

if __name__ == "__main__":
    main()
