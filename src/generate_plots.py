#!/usr/bin/env python3
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from utils.fragment_finder import analyze_formula_complexity

class ModalDepthCalculator:
    def __init__(self):
        self.memo = {}

    def get_modal_depth(self, formula: str) -> int:
        formula = formula.strip()

        if not formula:  # Base case for an empty subformula
            return 0
        
        if formula in self.memo: # Return cached result if available
            return self.memo[formula]

        # 1. Handle outermost balanced parentheses: (A)
        # Check if the formula is enclosed in a single pair of parentheses that span the entire formula.
        if formula.startswith('(') and formula.endswith(')'):
            open_paren_count = 0
            is_outermost_enclosing = False 
            # Check if the first '(' matches the last ')' without closing prematurely
            # For example, in (A) & B, the parentheses around A are not the outermost for the whole string.
            for i, char in enumerate(formula):
                if char == '(':
                    open_paren_count += 1
                elif char == ')':
                    open_paren_count -= 1
                
                if open_paren_count == 0: # Balanced point found
                    if i == len(formula) - 1: # If balance occurs at the very end, it's (A)
                        is_outermost_enclosing = True
                    # If balance occurs before the end, it's like (A)opB, so not a simple (A) structure.
                    break 
            
            if is_outermost_enclosing:
                # Recursively call on the content inside the parentheses
                result = self.get_modal_depth(formula[1:-1])
                self.memo[formula] = result
                return result

        # 2. Handle binary operators: A -> B, A & B, A | B
        # We scan from right to left. The outermost, lowest-precedence operator is processed.
        # Precedence order for splitting: -> (lowest), then & and | (same level).
        
        paren_level = 0
        # Try to split by '->' (implication)
        # Iterate from right to left to find the main implication connective
        for i in range(len(formula) - 1, -1, -1):
            char = formula[i]
            if char == ')':
                paren_level += 1
            elif char == '(':
                paren_level -= 1
            # Check for '->' operator (length 2)
            # Ensure 'i' is not at the beginning to avoid index out of bounds for formula[i-1]
            elif i > 0 and formula[i-1:i+1] == '->' and paren_level == 0:
                left_subformula = formula[:i-1]
                right_subformula = formula[i+1:]
                result = max(self.get_modal_depth(left_subformula), self.get_modal_depth(right_subformula))
                self.memo[formula] = result
                return result
        
        paren_level = 0 # Reset for next operator type
        # Try to split by '&' (conjunction) or '|' (disjunction)
        # These are often at the same precedence level.
        for i in range(len(formula) - 1, -1, -1):
            char = formula[i]
            if char == ')':
                paren_level += 1
            elif char == '(':
                paren_level -= 1
            elif (char == '&' or char == '|') and paren_level == 0:
                left_subformula = formula[:i]
                right_subformula = formula[i+1:]
                result = max(self.get_modal_depth(left_subformula), self.get_modal_depth(right_subformula))
                self.memo[formula] = result
                return result

        # 3. Handle unary operator: !A (negation)
        if formula.startswith('!'):
            sub_formula = formula[1:]
            result = self.get_modal_depth(sub_formula) # Negation does not add to modal depth
            self.memo[formula] = result
            return result

        # 4. Handle modal operators: <op_name>A or [op_name]A
        if formula.startswith('<') or formula.startswith('['):
            open_bracket_char = formula[0]
            close_bracket_char = '>' if open_bracket_char == '<' else ']'
            
            operator_name_end_idx = -1
            try:
                # Find the first closing bracket that terminates the operator name part
                # e.g., in "<before>p", this finds the index of '>' after "before".
                # This assumes operator names (like "before", "overlapped_by") do not contain
                # the closing bracket character ('>' or ']') themselves.
                operator_name_end_idx = formula.index(close_bracket_char)
            except ValueError:
                # This means the formula starts with '<' or '[' but no corresponding '>' or ']' was found.
                # This could be a malformed formula or a proposition that happens to start with these characters.
                # In such a case, it will fall through to be treated as a proposition (depth 0).
                pass 

            if operator_name_end_idx != -1:
                # The subformula is everything after the operator_name's closing bracket
                sub_formula_str = formula[operator_name_end_idx+1:]
                
                # Modal operator increases depth by 1 + depth of its subformula
                result = 1 + self.get_modal_depth(sub_formula_str)
                self.memo[formula] = result
                return result
            # If operator_name_end_idx was -1, it's not a recognized modal operator here.

        # 5. Base case: Propositional variable (or an unparsed/malformed segment)
        # If none of the above rules matched, it's assumed to be a proposition.
        # Propositional variables have a modal depth of 0.
        self.memo[formula] = 0
        return 0

def calculate_modal_depth(formula: str) -> int:
    """
    A wrapper function to create a ModalDepthCalculator instance and calculate modal depth.
    
    Args:
        formula: The temporal logic formula string.

    Returns:
        The modal depth of the formula.
    """
    calculator = ModalDepthCalculator()
    return calculator.get_modal_depth(formula)

# --- Confidence Interval Calculation ---
Z_SCORE = 1.96 # For 95% confidence interval

def calculate_normal_approximation_interval(p, n, z=Z_SCORE):
    """Calculates the Wald interval for a binomial proportion."""
    if n == 0 or pd.isna(p) or not (0 <= p <= 1):
        return (0.0, 0.0)
    if p == 0 or p == 1: # Avoid math domain error for sqrt when p(1-p) is 0
        standard_error = 0
    else:
        standard_error = math.sqrt((p * (1 - p)) / n)
    
    lower = max(0.0, p - z * standard_error)
    upper = min(1.0, p + z * standard_error)
    return lower, upper

def transform_model_name(model_name):
    """Transforms model names like "model-chat-v1" to "Model Chat V1"."""
    return ' '.join(word.capitalize() for word in model_name.replace('-', ' ').split())

def parse_filename(filename):
    """Parses model and config name from filename "model>-config.json"."""
    base = filename.replace(".json", "")
    parts = base.split(">-")
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    print(f"Warning: Filename '{filename}' not parsed. Skipping.", file=sys.stderr)
    return None, None

def transform_config_name(old_config_name):
    """Transforms config names like '6shot-ctx-cot' to 'ctx+cot+fs'."""
    components = []
    low_name = old_config_name.lower()
    if "ctx" in low_name: components.append("ctx")
    if "cot" in low_name: components.append("cot")
    if "shot" in low_name: components.append("fs") # 'shot' maps to 'fs'
    return "+".join(sorted(components)) if components else old_config_name

def _sanitize_filename(name_part):
    """Helper to make names safe for filenames."""
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name_part)

def balance_df_within_bins(input_df, bin_group_columns, type_column='formula_type', type1='VALID', type2='UNSATISFIABLE', random_state=42):
    """Balances samples for type1 and type2 within each bin group."""
    if input_df.empty: return pd.DataFrame(columns=input_df.columns)
    
    balanced_dfs = []
    # Ensure bin_group_columns is a list, even if a single string is passed
    if isinstance(bin_group_columns, str):
        bin_group_columns = [bin_group_columns]
        
    grouping_cols = ['model_name', 'config_name'] + bin_group_columns
    
    missing_cols = [col for col in grouping_cols if col not in input_df.columns]
    if missing_cols:
        print(f"Error balancing: Missing columns {missing_cols} in input DataFrame.", file=sys.stderr)
        # Attempt to proceed if only bin_group_columns are missing, but warn.
        # If model_name or config_name are missing, it's more critical.
        if 'model_name' in missing_cols or 'config_name' in missing_cols:
             print("Critical columns model_name or config_name missing for balancing. Returning empty DataFrame.", file=sys.stderr)
             return pd.DataFrame(columns=input_df.columns)
        # Filter out missing bin_group_columns from grouping_cols if they don't exist
        grouping_cols = [col for col in grouping_cols if col in input_df.columns]
        if not any(col in bin_group_columns for col in grouping_cols): # if all bin_group_columns were missing
            print(f"Warning: All specified bin_group_columns {bin_group_columns} are missing. Balancing will be broader.", file=sys.stderr)


    for _, group in input_df.groupby(grouping_cols, observed=True, dropna=False):
        type1_samples = group[group[type_column] == type1]
        type2_samples = group[group[type_column] == type2]
        min_n = min(len(type1_samples), len(type2_samples))
        if min_n > 0:
            balanced_dfs.append(type1_samples.sample(n=min_n, random_state=random_state))
            balanced_dfs.append(type2_samples.sample(n=min_n, random_state=random_state))
            
    return pd.concat(balanced_dfs).reset_index(drop=True) if balanced_dfs else pd.DataFrame(columns=input_df.columns)

def load_and_process_data(directory, size_window, depth_window, min_samples, output_dir, use_latex):
    """Loads, processes, balances data, and triggers distribution plots."""
    all_results = []
    print(f"Scanning directory: {directory}")
    try:
        filenames = [f for f in os.listdir(directory) if f.endswith(".json")]
    except Exception as e:
        print(f"Error accessing directory {directory}: {e}", file=sys.stderr)
        return None, None, None

    if not filenames:
        print(f"Warning: No JSON files in {directory}", file=sys.stderr)
        return None, None, None

    print(f"Found {len(filenames)} JSON files.")
    for filename in filenames:
        model_name_raw, config_name_raw = parse_filename(filename)
        if not model_name_raw or not config_name_raw: continue
        
        model_name = transform_model_name(model_name_raw)
        config_name = transform_config_name(config_name_raw)
        filepath = os.path.join(directory, filename)

        try:
            with open(filepath, 'r') as f: data = json.load(f)
            raw_results = data.get("results", {}).get("raw_results", [])
            if not isinstance(raw_results, list): continue

            for item in raw_results:
                formula_info = item.get("formula", {})
                formula_size = formula_info.get("size")
                formula_type = formula_info.get("type")
                is_correct = item.get("response_is_correct")
                formula_text = formula_info.get("formula")

                if not all(v is not None for v in [formula_size, formula_type, is_correct, formula_text]): continue
                if formula_type not in ['VALID', 'UNSATISFIABLE']: continue # Only these types for balancing

                try:
                    modal_depth = calculate_modal_depth(formula_text)
                except Exception: 
                    modal_depth = 0
                
                # Calculate complexity
                try:
                    complexity = analyze_formula_complexity(formula_text)
                    complexity_name = complexity.name
                except Exception:
                    complexity_name = "UNKNOWN"
                
                all_results.append({
                    "model_name": model_name, "config_name": config_name,
                    "formula_size": int(formula_size), "formula_type": formula_type,
                    "modal_depth": modal_depth, "is_correct": bool(is_correct),
                    "complexity_class": complexity_name,
                    "original_model_name": model_name_raw 
                })
        except Exception as e:
            print(f"Warning: Error processing {filename}: {e}. Skipping.", file=sys.stderr)

    if not all_results:
        print("Error: No valid 'VALID' or 'UNSATISFIABLE' results found.", file=sys.stderr)
        return None, None, None
    
    df = pd.DataFrame(all_results)
    if df.empty: return None, None, None

    # --- Size-based Analysis & Distribution Plot ---
    df_for_size = df.copy()
    df_for_size['size_group_val'] = (df_for_size['formula_size'] // size_window) * size_window
    df_for_size['size_group_label'] = df_for_size['size_group_val'].apply(lambda x: f"{x}-{x + size_window - 1}")
    
    print("Balancing samples within size bins (for accuracy plots)...")
    # Pass 'size_group_label' as a list to balance_df_within_bins
    df_balanced_size = balance_df_within_bins(df_for_size, ['size_group_label'])


    unique_models_raw_names = df['original_model_name'].unique() 
    for model_raw_name_iter in unique_models_raw_names:
        model_display_name = transform_model_name(model_raw_name_iter)
        
        # Filter data for the current model BEFORE passing to plotting function
        current_model_df_for_size = df_for_size[df_for_size['original_model_name'] == model_raw_name_iter]
        current_model_df_balanced_size = df_balanced_size[df_balanced_size['original_model_name'] == model_raw_name_iter]

        if not current_model_df_for_size.empty: # Plot only if there's original data for this model
            plot_distribution_before_after_balancing(
                current_model_df_for_size, 
                current_model_df_balanced_size, # This might be empty if no balancing occurred for this model's bins
                model_display_name, 
                _sanitize_filename(model_raw_name_iter), 
                'size_group_label', 'Formula Size Bin',
                output_dir, "size_dist", use_latex
            )
    
    empty_agg_df_cols = ['model_name', 'config_name', 'size_group_val', 'size_group_label', 'original_model_name', 'n', 'correct_count', 'accuracy', 'ci_lower', 'ci_upper', 'total_n_in_bin']
    size_agg = pd.DataFrame(columns=empty_agg_df_cols) # Ensure original_model_name is in columns
    if not df_balanced_size.empty:
        # Ensure 'original_model_name' is part of groupby if it's used in later merges or references
        size_grouped = df_balanced_size.groupby(['model_name', 'config_name', 'size_group_val', 'size_group_label', 'original_model_name'], observed=True, dropna=False)
        size_agg_temp = size_grouped['is_correct'].agg(n='count', correct_count='sum').reset_index()
        size_agg_temp = size_agg_temp[size_agg_temp['n'] >= min_samples]
        if not size_agg_temp.empty:
            size_agg_temp['accuracy'] = np.where(size_agg_temp['n'] > 0, size_agg_temp['correct_count'] / size_agg_temp['n'], 0)
            ci_bounds = size_agg_temp.apply(lambda r: calculate_normal_approximation_interval(r['accuracy'], r['n']), axis=1)
            size_agg_temp[['ci_lower', 'ci_upper']] = pd.DataFrame(ci_bounds.tolist(), index=size_agg_temp.index)
            
            size_samples_balanced = df_balanced_size.groupby(['model_name', 'size_group_label', 'original_model_name'], observed=True, dropna=False)['is_correct'].agg(total_n_in_bin='count').reset_index()
            size_agg = pd.merge(size_agg_temp, size_samples_balanced, on=['model_name', 'size_group_label', 'original_model_name'], how='left')


    # --- Depth-based Analysis & Distribution Plot ---
    df_for_depth = df.copy()
    df_for_depth['depth_group_val'] = (df_for_depth['modal_depth'] // depth_window) * depth_window
    df_for_depth['depth_group_label'] = df_for_depth['depth_group_val'].apply(lambda x: f"{x}-{x + depth_window - 1}")

    print("Balancing samples within depth bins (for accuracy plots)...")
    df_balanced_depth = balance_df_within_bins(df_for_depth, ['depth_group_label'])

    for model_raw_name_iter in unique_models_raw_names:
        model_display_name = transform_model_name(model_raw_name_iter)
        current_model_df_for_depth = df_for_depth[df_for_depth['original_model_name'] == model_raw_name_iter]
        current_model_df_balanced_depth = df_balanced_depth[df_balanced_depth['original_model_name'] == model_raw_name_iter]

        if not current_model_df_for_depth.empty:
            plot_distribution_before_after_balancing(
                current_model_df_for_depth,
                current_model_df_balanced_depth,
                model_display_name,
                _sanitize_filename(model_raw_name_iter),
                'depth_group_label', 'Modal Depth Bin',
                output_dir, "depth_dist", use_latex
            )

    depth_agg = pd.DataFrame(columns=empty_agg_df_cols) 
    if not df_balanced_depth.empty:
        depth_grouped = df_balanced_depth.groupby(['model_name', 'config_name', 'depth_group_val', 'depth_group_label', 'original_model_name'], observed=True, dropna=False)
        depth_agg_temp = depth_grouped['is_correct'].agg(n='count', correct_count='sum').reset_index()
        depth_agg_temp = depth_agg_temp[depth_agg_temp['n'] >= min_samples]
        if not depth_agg_temp.empty:
            depth_agg_temp['accuracy'] = np.where(depth_agg_temp['n'] > 0, depth_agg_temp['correct_count'] / depth_agg_temp['n'], 0)
            ci_bounds_d = depth_agg_temp.apply(lambda r: calculate_normal_approximation_interval(r['accuracy'], r['n']), axis=1)
            depth_agg_temp[['ci_lower', 'ci_upper']] = pd.DataFrame(ci_bounds_d.tolist(), index=depth_agg_temp.index)
            depth_samples_balanced = df_balanced_depth.groupby(['model_name', 'depth_group_label', 'original_model_name'], observed=True, dropna=False)['is_correct'].agg(total_n_in_bin='count').reset_index()
            depth_agg = pd.merge(depth_agg_temp, depth_samples_balanced, on=['model_name', 'depth_group_label', 'original_model_name'], how='left')
    
    # --- Complexity-based Analysis ---
    df_for_complexity = df.copy()
    print("Balancing samples within complexity classes (for accuracy plots)...")
    df_balanced_complexity = balance_df_within_bins(df_for_complexity, ['complexity_class'])
    
    # Plot complexity distribution for each model
    for model_raw_name_iter in unique_models_raw_names:
        model_display_name = transform_model_name(model_raw_name_iter)
        current_model_df_for_complexity = df_for_complexity[df_for_complexity['original_model_name'] == model_raw_name_iter]
        current_model_df_balanced_complexity = df_balanced_complexity[df_balanced_complexity['original_model_name'] == model_raw_name_iter]

        if not current_model_df_for_complexity.empty:
            plot_distribution_before_after_balancing(
                current_model_df_for_complexity,
                current_model_df_balanced_complexity,
                model_display_name,
                _sanitize_filename(model_raw_name_iter),
                'complexity_class', 'Complexity Class',
                output_dir, "complexity_dist", use_latex
            )
    
    # Aggregate complexity data
    empty_complexity_agg_cols = ['model_name', 'config_name', 'complexity_class', 'original_model_name', 'n', 'correct_count', 'accuracy', 'ci_lower', 'ci_upper', 'total_n_in_class']
    complexity_agg = pd.DataFrame(columns=empty_complexity_agg_cols)
    if not df_balanced_complexity.empty:
        complexity_grouped = df_balanced_complexity.groupby(['model_name', 'config_name', 'complexity_class', 'original_model_name'], observed=True, dropna=False)
        complexity_agg_temp = complexity_grouped['is_correct'].agg(n='count', correct_count='sum').reset_index()
        complexity_agg_temp = complexity_agg_temp[complexity_agg_temp['n'] >= min_samples]
        if not complexity_agg_temp.empty:
            complexity_agg_temp['accuracy'] = np.where(complexity_agg_temp['n'] > 0, complexity_agg_temp['correct_count'] / complexity_agg_temp['n'], 0)
            ci_bounds_c = complexity_agg_temp.apply(lambda r: calculate_normal_approximation_interval(r['accuracy'], r['n']), axis=1)
            complexity_agg_temp[['ci_lower', 'ci_upper']] = pd.DataFrame(ci_bounds_c.tolist(), index=complexity_agg_temp.index)
            complexity_samples_balanced = df_balanced_complexity.groupby(['model_name', 'complexity_class', 'original_model_name'], observed=True, dropna=False)['is_correct'].agg(total_n_in_class='count').reset_index()
            complexity_agg = pd.merge(complexity_agg_temp, complexity_samples_balanced, on=['model_name', 'complexity_class', 'original_model_name'], how='left')
            
    print("\nData processing complete.")
    return (
        size_agg.sort_values(by=['model_name', 'config_name', 'size_group_val']) if not size_agg.empty else pd.DataFrame(columns=size_agg.columns),
        depth_agg.sort_values(by=['model_name', 'config_name', 'depth_group_val']) if not depth_agg.empty else pd.DataFrame(columns=depth_agg.columns),
        complexity_agg.sort_values(by=['model_name', 'config_name', 'complexity_class']) if not complexity_agg.empty else pd.DataFrame(columns=complexity_agg.columns)
    )

# --- Plotting Functions ---
def plot_distribution_before_after_balancing(
    df_original_model_binned, df_balanced_model_binned,
    model_display_name, model_filename_part,
    bin_col, bin_name_str,
    output_dir, file_suffix, use_latex=True
):
    """Plots VALID/UNSATISFIABLE distribution before and after balancing for a single model."""
    
    # Aggregate counts for 'VALID' and 'UNSATISFIABLE' before balancing
    # Ensure bin_col exists before groupby
    if bin_col not in df_original_model_binned.columns:
        print(f"Warning: bin_col '{bin_col}' not found in df_original_model_binned for {model_display_name}. Skipping distribution plot.", file=sys.stderr)
        return
        
    counts_before = df_original_model_binned.groupby([bin_col, 'formula_type'], observed=True).size().unstack(fill_value=0)
    for type_val in ['VALID', 'UNSATISFIABLE']: 
        if type_val not in counts_before: counts_before[type_val] = 0
    counts_before = counts_before.rename(columns={'VALID': 'VALID_before', 'UNSATISFIABLE': 'UNSATISFIABLE_before'})
    
    # Aggregate counts after balancing
    # df_balanced_model_binned might be empty if no samples for this model were balanced
    if not df_balanced_model_binned.empty and bin_col in df_balanced_model_binned.columns:
        counts_after = df_balanced_model_binned.groupby([bin_col, 'formula_type'], observed=True).size().unstack(fill_value=0)
        for type_val in ['VALID', 'UNSATISFIABLE']: 
            if type_val not in counts_after: counts_after[type_val] = 0
        counts_after = counts_after.rename(columns={'VALID': 'VALID_after', 'UNSATISFIABLE': 'UNSATISFIABLE_after'})
    else: # Create an empty counts_after with the same index as counts_before if balanced data is empty
        counts_after = pd.DataFrame(index=counts_before.index, columns=['VALID_after', 'UNSATISFIABLE_after']).fillna(0)


    # Prepare for merge by selecting only the relevant columns
    # This also handles cases where one of the types (VALID/UNSATISFIABLE) might be missing entirely
    cols_before_selected = [col for col in ['VALID_before', 'UNSATISFIABLE_before'] if col in counts_before.columns]
    counts_before_for_merge = counts_before[cols_before_selected] if cols_before_selected else pd.DataFrame(index=counts_before.index)

    cols_after_selected = [col for col in ['VALID_after', 'UNSATISFIABLE_after'] if col in counts_after.columns]
    counts_after_for_merge = counts_after[cols_after_selected] if cols_after_selected else pd.DataFrame(index=counts_after.index)

    # Merge before and after counts using index
    plot_data = pd.merge(
        counts_before_for_merge,
        counts_after_for_merge,
        left_index=True,  # Merge on the index (bin_col)
        right_index=True, # Merge on the index (bin_col)
        how='outer'
    ).fillna(0).reset_index() # reset_index() makes the index (bin_col) a column

    if plot_data.empty or not any(col.endswith("_before") or col.endswith("_after") for col in plot_data.columns):
        # The 'bin_col' itself might exist from reset_index even if no data columns.
        # So, check for data columns specifically.
        data_cols_present = any(col.startswith("VALID_") or col.startswith("UNSATISFIABLE_") for col in plot_data.columns)
        if not data_cols_present and not plot_data[bin_col].dropna().empty: # If only bin_col is present with values
             pass # Allow plotting if at least bins are there, bars will be zero
        elif not data_cols_present : # Truly no data or bins
            print(f"No data to plot for distribution: {model_display_name}, {bin_name_str}", file=sys.stderr)
            return


    # Ensure all expected data columns are present after merge, fill with 0 if not
    for col_suffix in ['_before', '_after']:
        for type_prefix in ['VALID', 'UNSATISFIABLE']:
            col_name = type_prefix + col_suffix
            if col_name not in plot_data:
                plot_data[col_name] = 0
    
    # Sort bins: "0-9", "10-19", etc. or complexity class order
    if bin_col not in plot_data.columns:
        print(f"Warning: bin_col '{bin_col}' is not in plot_data columns after merge for {model_display_name}. Cannot sort for plot.", file=sys.stderr)
    else:
        # Ensure plot_data[bin_col] does not contain NaN before trying to split, or handle it
        plot_data = plot_data.dropna(subset=[bin_col])
        if not plot_data.empty:
            if bin_col == 'complexity_class':
                # Define complexity order
                complexity_order = ['NP_COMPLETE', 'NEXPTIME_COMPLETE', 'EXP_SPACE_COMPLETE', 'NON_PRIMITIVE_RECURSIVE', 'UNDECIDABLE', 'UNKNOWN']
                plot_data['sort_key'] = plot_data[bin_col].apply(lambda x: complexity_order.index(x) if x in complexity_order else len(complexity_order))
            else:
                plot_data['sort_key'] = plot_data[bin_col].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) and '-' in x else -1)
            plot_data = plot_data.sort_values(by='sort_key').reset_index(drop=True)
        else:
            print(f"Warning: plot_data became empty after dropna on bin_col for {model_display_name}", file=sys.stderr)
            return


    plt.rcParams.update({
        "text.usetex": use_latex, "font.family": "serif" if use_latex else "sans-serif",
        "font.serif": ["Computer Modern Roman"] if use_latex else ["DejaVu Serif"],
        "axes.labelsize": 11, "font.size": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.figsize": (10, 6), # Adjusted for potentially more bins
    })
    if use_latex: plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
    
    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray', zorder=0)
    ax.set_axisbelow(True)

    # Check if plot_data has the bin_col after all processing, if not, cannot proceed with plotting this specific chart
    if bin_col not in plot_data.columns or plot_data.empty:
        print(f"Final check: No data or bin column '{bin_col}' in plot_data for {model_display_name}. Skipping this distribution chart.", file=sys.stderr)
        plt.close(fig) # Close the figure if we can't plot
        return

    n_groups = len(plot_data[bin_col])
    if n_groups == 0:
        print(f"No groups to plot for distribution: {model_display_name}, {bin_name_str}", file=sys.stderr)
        plt.close(fig)
        return
        
    ind = np.arange(n_groups) 
    width = 0.20 # Width of the bars

    # Plotting bars
    ax.bar(ind - 1.5*width, plot_data['VALID_before'], width, label='Valid (Before)', color='skyblue', zorder=3)
    ax.bar(ind - 0.5*width, plot_data['UNSATISFIABLE_before'], width, label='Unsatisfiable (Before)', color='lightcoral', zorder=3)
    ax.bar(ind + 0.5*width, plot_data['VALID_after'], width, label='Valid (After)', color='steelblue', hatch='//', zorder=3)
    ax.bar(ind + 1.5*width, plot_data['UNSATISFIABLE_after'], width, label='Unsatisfiable (After)', color='indianred', hatch='\\\\', zorder=3)

    ax.set_ylabel('Number of Formulas')
    ax.set_xlabel(bin_name_str)
    
    title_model_part = model_display_name.replace("_", r"\_") if use_latex else model_display_name
    title_main = r"\textbf{Formula Distribution Before \& After Balancing}" if use_latex else "Formula Distribution Before & After Balancing"
    title = f"{title_main}\nModel: {title_model_part} ({bin_name_str})"
    ax.set_title(title, pad=15)

    ax.set_xticks(ind)
    
    # Special handling for complexity class labels
    if bin_col == 'complexity_class':
        # Create more readable labels
        label_map = {
            'NP_COMPLETE': 'NP-Complete',
            'NEXPTIME_COMPLETE': 'NEXPTIME-Complete',
            'EXP_SPACE_COMPLETE': 'EXPSPACE-Complete',
            'NON_PRIMITIVE_RECURSIVE': 'Non-Primitive Recursive',
            'UNDECIDABLE': 'Undecidable',
            'UNKNOWN': 'Unknown'
        }
        plot_labels = [label_map.get(x, x) for x in plot_data[bin_col]]
        ax.set_xticklabels(plot_labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels(plot_data[bin_col], rotation=45, ha='right') # Increased rotation for better label visibility
    
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to prevent title/label cutoff

    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{model_filename_part}_{file_suffix}"
    plot_filename_pdf = os.path.join(output_dir, f"{base_filename}.pdf")
    plot_filename_png = os.path.join(output_dir, f"{base_filename}.png")
    
    try:
        plt.savefig(plot_filename_pdf, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved: {plot_filename_pdf} and {plot_filename_png}")
    except Exception as e:
        print(f"Error saving distribution plot {plot_filename_pdf}: {e}", file=sys.stderr)
    plt.close(fig)


def plot_model_accuracy(model_data, model_name, output_dir, size_window, min_samples=5, use_latex=True):
    """Plots accuracy vs. formula size for a single model."""
    if model_data.empty or 'total_n_in_bin' not in model_data.columns:
        print(f"Skipping accuracy plot for {model_name} (size): No data or 'total_n_in_bin' missing.", file=sys.stderr)
        return

    plt.rcParams.update({
        "text.usetex": use_latex, "font.family": "serif" if use_latex else "sans-serif",
        "font.serif": ["Computer Modern Roman"] if use_latex else ["DejaVu Serif"],
        "axes.labelsize": 11, "font.size": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.figsize": (7, 5),
    })
    if use_latex: plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
    
    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray', zorder=0)
    ax.set_axisbelow(True)

    configs = model_data['config_name'].unique()
    def config_complexity_score(cn):
        s=0; comps=cn.lower().split('+')
        if any(x in comps for x in ["barebone", "base"]): return 0 if "barebone" in comps else 1
        if "ctx" in comps: s+=3
        if "cot" in comps: s+=2
        if "fs" in comps: s+=1
        return s+2
    
    sorted_configs = sorted(configs, key=config_complexity_score, reverse=True)
    palette = plt.cm.viridis(np.linspace(0,1,len(sorted_configs)))

    unique_labels = model_data['size_group_label'].dropna().unique()
    if not len(unique_labels): # Check if unique_labels is empty
        print(f"No size group labels to plot for accuracy: {model_name}", file=sys.stderr)
        plt.close(fig)
        return
    all_size_labels = sorted(unique_labels, key=lambda x: int(x.split('-')[0]))
    x_indices = np.arange(len(all_size_labels))
    
    # Ensure total_n_in_bin exists before groupby
    if 'total_n_in_bin' not in model_data.columns:
        print(f"Warning: 'total_n_in_bin' missing in model_data for {model_name} (size accuracy). Plot N labels might be incorrect.", file=sys.stderr)
        total_n_per_bin = {label: 0 for label in all_size_labels} # Default to 0
    else:
        total_n_per_bin = model_data.groupby('size_group_label', observed=True)['total_n_in_bin'].first().fillna(0).astype(int).to_dict()
    
    plot_labels_x = [(f"{lbl} $(N={total_n_per_bin.get(lbl,0)})$" if use_latex else f"{lbl} (N={total_n_per_bin.get(lbl,0)})") for lbl in all_size_labels]

    for i, config in enumerate(sorted_configs):
        # Ensure 'size_group_label' is in model_data before set_index
        if 'size_group_label' not in model_data.columns:
            print(f"Warning: 'size_group_label' missing for config {config} in {model_name}. Skipping this line in plot.", file=sys.stderr)
            continue
        cd = model_data[model_data['config_name'] == config].set_index('size_group_label').reindex(all_size_labels).reset_index()
        
        # Ensure 'size_group_label' is present after reindex and reset
        if 'size_group_label' not in cd.columns: continue

        cd = cd.dropna(subset=['size_group_label']) # Drop rows where size_group_label might be NaN after reindex
        if cd.empty : continue

        cd['sort_key'] = cd['size_group_label'].apply(lambda x: int(x.split('-')[0])) # x is already string
        cd = cd.sort_values(by='sort_key').reset_index(drop=True)
        
        valid_idx = cd['accuracy'].notna()
        # Ensure all_size_labels has content before list comprehension for px
        if not all_size_labels: continue
        px = [x_indices[all_size_labels.index(lbl)] for lbl in cd.loc[valid_idx, 'size_group_label'] if lbl in all_size_labels]
        
        if not cd.loc[valid_idx].empty:
            p_acc, p_cil, p_ciu = cd.loc[valid_idx, 'accuracy'].values, cd.loc[valid_idx, 'ci_lower'].values, cd.loc[valid_idx, 'ci_upper'].values
        else: # Handle empty selection
            p_acc, p_cil, p_ciu = [], [], []

        marker = ['o','s','^','D','v','<','>','p','*'][i % 9]
        if len(px) > 0:
            ax.plot(px, p_acc, label=config, marker=marker, color=palette[i], linewidth=1.5, markersize=4, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
            ax.fill_between(px, p_cil, p_ciu, color=palette[i], alpha=0.15, linewidth=0, zorder=2)

    xlabel = r"Formula Size Group ($N=\text{total samples in bin for model}$)" if use_latex else "Formula Size Group (N=total samples in bin for model)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Overall Accuracy (95\% CI)" if use_latex else "Overall Accuracy (95% CI)")
    
    model_disp = model_name.replace("_", r"\_") if use_latex else model_name
    title = (r"\textbf{" + model_disp + r"} \par \normalsize{Accuracy vs. Formula Size (Balanced, Window: " + str(size_window) + r", Min Samples per Config: " + str(min_samples) + r")}") if use_latex \
            else f"{model_disp}\nAccuracy vs. Formula Size (Balanced, Window: {size_window}, Min Samples per Config: {min_samples})"
    ax.set_title(title, pad=20 if use_latex else 10)

    if len(x_indices) > 0 : # Only set ticks if there are any
        ax.set_xticks(x_indices); ax.set_xticklabels(plot_labels_x, rotation=30, ha='right')
    ax.set_ylim(-0.02, 1.05); ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=11, prune='both'))
    ax.set_yticklabels([(f"${x:.1f}$" if use_latex else f"{x:.1f}") for x in ax.get_yticks()])
    
    leg_title = r"\textbf{Configuration}" if use_latex else "Configuration"
    leg = ax.legend(title=leg_title, bbox_to_anchor=(1.03,1), loc='upper left', frameon=True, framealpha=0.7)
    if use_latex: leg.get_title().set_fontsize(10)

    plt.tight_layout(rect=[0,0,0.85,0.95 if use_latex else 0.92])
    os.makedirs(output_dir, exist_ok=True)
    
    # Use original_model_name for filename if available and model_data is not empty
    safe_model_name_for_file = model_name # Default
    if not model_data.empty and 'original_model_name' in model_data.columns:
        first_original_name = model_data['original_model_name'].iloc[0]
        if pd.notna(first_original_name):
             safe_model_name_for_file = _sanitize_filename(first_original_name)

    base_fn = f"{safe_model_name_for_file}_size_accuracy_balanced"
    try:
        plt.savefig(os.path.join(output_dir, f"{base_fn}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{base_fn}.png"), dpi=300, bbox_inches='tight')
        print(f"Size accuracy plot saved: {output_dir}/{base_fn}.pdf/png")
    except Exception as e: print(f"Error saving size accuracy plot: {e}", file=sys.stderr)
    plt.close(fig)


def plot_model_accuracy_by_depth(model_data, model_name, output_dir, depth_window, min_samples=5, use_latex=True):
    """Plots accuracy vs. modal depth for a single model."""
    if model_data.empty or 'total_n_in_bin' not in model_data.columns:
        print(f"Skipping accuracy plot for {model_name} (depth): No data or 'total_n_in_bin' missing.", file=sys.stderr)
        return
    plt.rcParams.update({
        "text.usetex": use_latex, "font.family": "serif" if use_latex else "sans-serif",
        "font.serif": ["Computer Modern Roman"] if use_latex else ["DejaVu Serif"],
        "axes.labelsize": 11, "font.size": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.figsize": (7, 5),
    })
    if use_latex: plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray', zorder=0)
    ax.set_axisbelow(True)

    configs = model_data['config_name'].unique()
    def config_complexity_score(cn): 
        s=0; comps=cn.lower().split('+')
        if any(x in comps for x in ["barebone", "base"]): return 0 if "barebone" in comps else 1
        if "ctx" in comps: s+=3
        if "cot" in comps: s+=2
        if "fs" in comps: s+=1
        return s+2
    sorted_configs = sorted(configs, key=config_complexity_score, reverse=True)
    palette = plt.cm.viridis(np.linspace(0,1,len(sorted_configs)))

    unique_labels = model_data['depth_group_label'].dropna().unique()
    if not len(unique_labels):
        print(f"No depth group labels to plot for accuracy: {model_name}", file=sys.stderr)
        plt.close(fig)
        return
    all_depth_labels = sorted(unique_labels, key=lambda x: int(x.split('-')[0]))
    x_indices = np.arange(len(all_depth_labels))

    if 'total_n_in_bin' not in model_data.columns:
        print(f"Warning: 'total_n_in_bin' missing in model_data for {model_name} (depth accuracy). Plot N labels might be incorrect.", file=sys.stderr)
        total_n_per_bin = {label: 0 for label in all_depth_labels}
    else:
        total_n_per_bin = model_data.groupby('depth_group_label', observed=True)['total_n_in_bin'].first().fillna(0).astype(int).to_dict()
    
    plot_labels_x = [(f"{lbl} $(N={total_n_per_bin.get(lbl,0)})$" if use_latex else f"{lbl} (N={total_n_per_bin.get(lbl,0)})") for lbl in all_depth_labels]

    for i, config in enumerate(sorted_configs):
        if 'depth_group_label' not in model_data.columns:
            print(f"Warning: 'depth_group_label' missing for config {config} in {model_name}. Skipping this line in plot.", file=sys.stderr)
            continue
        cd = model_data[model_data['config_name'] == config].set_index('depth_group_label').reindex(all_depth_labels).reset_index()
        
        if 'depth_group_label' not in cd.columns: continue
        cd = cd.dropna(subset=['depth_group_label'])
        if cd.empty : continue
        
        cd['sort_key'] = cd['depth_group_label'].apply(lambda x: int(x.split('-')[0]))
        cd = cd.sort_values(by='sort_key').reset_index(drop=True)
        
        valid_idx = cd['accuracy'].notna()
        if not all_depth_labels: continue
        px = [x_indices[all_depth_labels.index(lbl)] for lbl in cd.loc[valid_idx, 'depth_group_label'] if lbl in all_depth_labels]
        
        if not cd.loc[valid_idx].empty:
            p_acc, p_cil, p_ciu = cd.loc[valid_idx, 'accuracy'].values, cd.loc[valid_idx, 'ci_lower'].values, cd.loc[valid_idx, 'ci_upper'].values
        else:
            p_acc, p_cil, p_ciu = [], [], []
        
        marker = ['o','s','^','D','v','<','>','p','*'][i % 9]
        if len(px) > 0:
            ax.plot(px, p_acc, label=config, marker=marker, color=palette[i], linewidth=1.5, markersize=4, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
            ax.fill_between(px, p_cil, p_ciu, color=palette[i], alpha=0.15, linewidth=0, zorder=2)

    xlabel = r"Modal Depth Group ($N=\text{total samples in bin for model}$)" if use_latex else "Modal Depth Group (N=total samples in bin for model)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Overall Accuracy (95\% CI)" if use_latex else "Overall Accuracy (95% CI)")

    model_disp = model_name.replace("_", r"\_") if use_latex else model_name
    title = (r"\textbf{" + model_disp + r"} \par \normalsize{Accuracy vs. Modal Depth (Balanced, Window: " + str(depth_window) + r", Min Samples per Config: " + str(min_samples) + r")}") if use_latex \
            else f"{model_disp}\nAccuracy vs. Modal Depth (Balanced, Window: {depth_window}, Min Samples per Config: {min_samples})"
    ax.set_title(title, pad=20 if use_latex else 10)

    if len(x_indices) > 0:
        ax.set_xticks(x_indices); ax.set_xticklabels(plot_labels_x, rotation=30, ha='right')
    ax.set_ylim(-0.02, 1.05); ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=11, prune='both'))
    ax.set_yticklabels([(f"${x:.1f}$" if use_latex else f"{x:.1f}") for x in ax.get_yticks()])

    leg_title = r"\textbf{Configuration}" if use_latex else "Configuration"
    leg = ax.legend(title=leg_title, bbox_to_anchor=(1.03,1), loc='upper left', frameon=True, framealpha=0.7)
    if use_latex: leg.get_title().set_fontsize(10)

    plt.tight_layout(rect=[0,0,0.85,0.95 if use_latex else 0.92])
    os.makedirs(output_dir, exist_ok=True)

    safe_model_name_for_file = model_name # Default
    if not model_data.empty and 'original_model_name' in model_data.columns:
        first_original_name = model_data['original_model_name'].iloc[0]
        if pd.notna(first_original_name):
            safe_model_name_for_file = _sanitize_filename(first_original_name)
    
    base_fn = f"{safe_model_name_for_file}_depth_accuracy_balanced"
    try:
        plt.savefig(os.path.join(output_dir, f"{base_fn}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{base_fn}.png"), dpi=300, bbox_inches='tight')
        print(f"Depth accuracy plot saved: {output_dir}/{base_fn}.pdf/png")
    except Exception as e: print(f"Error saving depth accuracy plot: {e}", file=sys.stderr)
    plt.close(fig)


def plot_model_accuracy_by_complexity(model_data, model_name, output_dir, min_samples=5, use_latex=True):
    """Plots accuracy vs. complexity class for a single model."""
    if model_data.empty or 'total_n_in_class' not in model_data.columns:
        print(f"Skipping accuracy plot for {model_name} (complexity): No data or 'total_n_in_class' missing.", file=sys.stderr)
        return
    
    plt.rcParams.update({
        "text.usetex": use_latex, "font.family": "serif" if use_latex else "sans-serif",
        "font.serif": ["Computer Modern Roman"] if use_latex else ["DejaVu Serif"],
        "axes.labelsize": 11, "font.size": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.figsize": (8, 5),
    })
    if use_latex: plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"

    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray', zorder=0)
    ax.set_axisbelow(True)

    configs = model_data['config_name'].unique()
    def config_complexity_score(cn): 
        s=0; comps=cn.lower().split('+')
        if any(x in comps for x in ["barebone", "base"]): return 0 if "barebone" in comps else 1
        if "ctx" in comps: s+=3
        if "cot" in comps: s+=2
        if "fs" in comps: s+=1
        return s+2
    sorted_configs = sorted(configs, key=config_complexity_score, reverse=True)
    palette = plt.cm.viridis(np.linspace(0,1,len(sorted_configs)))

    # Define complexity class order and labels
    complexity_order = ['NP_COMPLETE', 'NEXPTIME_COMPLETE', 'EXP_SPACE_COMPLETE', 'NON_PRIMITIVE_RECURSIVE', 'UNDECIDABLE', 'UNKNOWN']
    complexity_labels = {
        'NP_COMPLETE': 'NP-Complete',
        'NEXPTIME_COMPLETE': 'NEXPTIME-Complete',
        'EXP_SPACE_COMPLETE': 'EXPSPACE-Complete',
        'NON_PRIMITIVE_RECURSIVE': 'Non-Primitive\nRecursive',
        'UNDECIDABLE': 'Undecidable',
        'UNKNOWN': 'Unknown'
    }
    
    # Filter to only include complexity classes that exist in the data
    available_classes = [cc for cc in complexity_order if cc in model_data['complexity_class'].unique()]
    if not available_classes:
        print(f"No complexity classes to plot for {model_name}", file=sys.stderr)
        plt.close(fig)
        return
    
    x_indices = np.arange(len(available_classes))
    
    if 'total_n_in_class' not in model_data.columns:
        print(f"Warning: 'total_n_in_class' missing in model_data for {model_name} (complexity accuracy).", file=sys.stderr)
        total_n_per_class = {cc: 0 for cc in available_classes}
    else:
        total_n_per_class = model_data.groupby('complexity_class', observed=True)['total_n_in_class'].first().fillna(0).astype(int).to_dict()
    
    plot_labels_x = [(f"{complexity_labels[cc]} $(N={total_n_per_class.get(cc,0)})$" if use_latex else f"{complexity_labels[cc]} (N={total_n_per_class.get(cc,0)})") for cc in available_classes]

    for i, config in enumerate(sorted_configs):
        if 'complexity_class' not in model_data.columns:
            print(f"Warning: 'complexity_class' missing for config {config} in {model_name}.", file=sys.stderr)
            continue
        
        cd = model_data[model_data['config_name'] == config].set_index('complexity_class').reindex(available_classes).reset_index()
        
        if 'complexity_class' not in cd.columns: continue
        cd = cd.dropna(subset=['complexity_class'])
        if cd.empty: continue
        
        valid_idx = cd['accuracy'].notna()
        if not available_classes: continue
        px = [x_indices[available_classes.index(cc)] for cc in cd.loc[valid_idx, 'complexity_class'] if cc in available_classes]
        
        if not cd.loc[valid_idx].empty:
            p_acc, p_cil, p_ciu = cd.loc[valid_idx, 'accuracy'].values, cd.loc[valid_idx, 'ci_lower'].values, cd.loc[valid_idx, 'ci_upper'].values
        else:
            p_acc, p_cil, p_ciu = [], [], []
        
        marker = ['o','s','^','D','v','<','>','p','*'][i % 9]
        if len(px) > 0:
            ax.plot(px, p_acc, label=config, marker=marker, color=palette[i], linewidth=1.5, markersize=6, markeredgecolor='white', markeredgewidth=0.5, zorder=3)
            ax.fill_between(px, p_cil, p_ciu, color=palette[i], alpha=0.15, linewidth=0, zorder=2)

    xlabel = r"Complexity Class ($N=\text{total samples in class for model}$)" if use_latex else "Complexity Class (N=total samples in class for model)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Overall Accuracy (95\% CI)" if use_latex else "Overall Accuracy (95% CI)")

    model_disp = model_name.replace("_", r"\_") if use_latex else model_name
    title = (r"\textbf{" + model_disp + r"} \par \normalsize{Accuracy vs. Complexity Class (Balanced, Min Samples per Config: " + str(min_samples) + r")}") if use_latex \
            else f"{model_disp}\nAccuracy vs. Complexity Class (Balanced, Min Samples per Config: {min_samples})"
    ax.set_title(title, pad=20 if use_latex else 10)

    if len(x_indices) > 0:
        ax.set_xticks(x_indices)
        ax.set_xticklabels(plot_labels_x, rotation=45, ha='right')
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=11, prune='both'))
    ax.set_yticklabels([(f"${x:.1f}$" if use_latex else f"{x:.1f}") for x in ax.get_yticks()])

    leg_title = r"\textbf{Configuration}" if use_latex else "Configuration"
    leg = ax.legend(title=leg_title, bbox_to_anchor=(1.03,1), loc='upper left', frameon=True, framealpha=0.7)
    if use_latex: leg.get_title().set_fontsize(10)

    plt.tight_layout(rect=[0,0,0.85,0.95 if use_latex else 0.92])
    os.makedirs(output_dir, exist_ok=True)

    safe_model_name_for_file = model_name
    if not model_data.empty and 'original_model_name' in model_data.columns:
        first_original_name = model_data['original_model_name'].iloc[0]
        if pd.notna(first_original_name):
            safe_model_name_for_file = _sanitize_filename(first_original_name)
    
    base_fn = f"{safe_model_name_for_file}_complexity_accuracy_balanced"
    try:
        plt.savefig(os.path.join(output_dir, f"{base_fn}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f"{base_fn}.png"), dpi=300, bbox_inches='tight')
        print(f"Complexity accuracy plot saved: {output_dir}/{base_fn}.pdf/png")
    except Exception as e: 
        print(f"Error saving complexity accuracy plot: {e}", file=sys.stderr)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze model accuracy, balance samples, and generate plots including distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-d", "--directory", required=True, help="Directory with JSON results.")
    parser.add_argument("-w", "--size-window", type=int, default=10, help="Window for formula size grouping.")
    parser.add_argument("--depth-window", type=int, default=1, help="Window for modal depth grouping.")
    parser.add_argument("-o", "--output-dir", default="plots_balanced_dist", help="Directory for all plots.")
    parser.add_argument("--min-samples", type=int, default=10, help="Min balanced samples per config in bin for accuracy plots.")
    parser.add_argument("--no-latex", action="store_true", help="Disable LaTeX styling for plots.")
    parser.add_argument("--skip-size-accuracy", action="store_true", help="Skip size-based accuracy plots.")
    parser.add_argument("--skip-depth-accuracy", action="store_true", help="Skip depth-based accuracy plots.")
    parser.add_argument("--skip-complexity-accuracy", action="store_true", help="Skip complexity-based accuracy plots.")
    args = parser.parse_args()

    if any(w <= 0 for w in [args.size_window, args.depth_window, args.min_samples]):
        sys.exit("Error: Window sizes and min_samples must be positive.")

    use_latex_rendering = not args.no_latex
    if use_latex_rendering:
        try:
            plt.rcParams.update({"text.usetex": True})
            fig_test, ax_test = plt.subplots(figsize=(0.1,0.1)); ax_test.text(0.5,0.5,r"$\alpha$"); plt.close(fig_test)
            print("LaTeX rendering enabled.")
        except RuntimeError:
            print("Warning: LaTeX rendering failed. Using standard matplotlib rendering.", file=sys.stderr)
            use_latex_rendering = False
            plt.rcParams.update({"text.usetex": False})

    size_data_agg, depth_data_agg, complexity_data_agg = load_and_process_data(
        args.directory, args.size_window, args.depth_window, args.min_samples,
        args.output_dir, use_latex_rendering 
    )

    if size_data_agg is None or depth_data_agg is None or complexity_data_agg is None: 
        print("Exiting due to data loading/processing issues. Aggregated data is None.", file=sys.stderr)
        sys.exit(1)
        
    unique_models_for_accuracy = sorted(list(
        set(size_data_agg['model_name'].unique() if not size_data_agg.empty else []) |
        set(depth_data_agg['model_name'].unique() if not depth_data_agg.empty else []) |
        set(complexity_data_agg['model_name'].unique() if not complexity_data_agg.empty else [])
    ))

    if not unique_models_for_accuracy:
        print("No models found with sufficient data for accuracy plots after processing. Distribution plots might still be generated.", file=sys.stderr)
    else:
        print(f"\nGenerating accuracy plots for models: {', '.join(unique_models_for_accuracy)}")
        print(f"Accuracy plots will be in directory: {args.output_dir}")

    for model_name_disp in unique_models_for_accuracy: 
        print(f"\nProcessing accuracy plots for model: {model_name_disp}")
        
        if not args.skip_size_accuracy:
            model_s_data = size_data_agg[size_data_agg['model_name'] == model_name_disp] if not size_data_agg.empty else pd.DataFrame()
            if not model_s_data.empty:
                plot_model_accuracy(model_s_data, model_name_disp, args.output_dir, args.size_window, args.min_samples, use_latex_rendering)
            else:
                print(f"No size-based accuracy data to plot for model: {model_name_disp}")
        
        if not args.skip_depth_accuracy:
            model_d_data = depth_data_agg[depth_data_agg['model_name'] == model_name_disp] if not depth_data_agg.empty else pd.DataFrame()
            if not model_d_data.empty:
                plot_model_accuracy_by_depth(model_d_data, model_name_disp, args.output_dir, args.depth_window, args.min_samples, use_latex_rendering)
            else:
                print(f"No depth-based accuracy data to plot for model: {model_name_disp}")
        
        if not args.skip_complexity_accuracy:
            model_c_data = complexity_data_agg[complexity_data_agg['model_name'] == model_name_disp] if not complexity_data_agg.empty else pd.DataFrame()
            if not model_c_data.empty:
                plot_model_accuracy_by_complexity(model_c_data, model_name_disp, args.output_dir, args.min_samples, use_latex_rendering)
            else:
                print(f"No complexity-based accuracy data to plot for model: {model_name_disp}")
                
    print("\nAnalysis complete. Check output directory for plots.")

if __name__ == "__main__":
    main()
