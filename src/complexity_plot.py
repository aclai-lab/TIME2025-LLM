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

Z_SCORE = 1.96  # 95% confidence interval

def calculate_normal_approximation_interval(p, n, z=Z_SCORE):
    if n == 0 or pd.isna(p) or not (0 <= p <= 1):
        return (0.0, 0.0)
    if p == 0 or p == 1:  # Avoid math domain error for sqrt when p(1-p) is 0
        standard_error = 0
    else:
        standard_error = math.sqrt((p * (1 - p)) / n)
    
    lower = max(0.0, p - z * standard_error)
    upper = min(1.0, p + z * standard_error)
    return lower, upper

def load_and_analyze_json(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        raw_results = data.get("results", {}).get("raw_results", [])
        if not isinstance(raw_results, list):
            print(f"Error: No valid raw_results found in {filepath}", file=sys.stderr)
            return None
        
        # Process each result
        results = []
        for item in raw_results:
            formula_info = item.get("formula", {})
            formula_text = formula_info.get("formula")
            is_correct = item.get("response_is_correct")
            
            if formula_text is None or is_correct is None:
                continue
            
            # Analyze complexity
            try:
                complexity = analyze_formula_complexity(formula_text)
                complexity_name = complexity.name
            except Exception as e:
                print(f"Warning: Error analyzing formula complexity: {e}", file=sys.stderr)
                complexity_name = "UNKNOWN"
            
            results.append({
                "formula": formula_text,
                "is_correct": bool(is_correct),
                "complexity_class": complexity_name
            })
        
        return results
    
    except Exception as e:
        print(f"Error loading file {filepath}: {e}", file=sys.stderr)
        return None

def calculate_accuracy_by_complexity(results):
    df = pd.DataFrame(results)
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by complexity class and calculate statistics
    grouped = df.groupby('complexity_class')
    stats = grouped['is_correct'].agg(['count', 'sum', 'mean']).reset_index()
    stats.columns = ['complexity_class', 'n', 'correct_count', 'accuracy']
    
    # Calculate confidence intervals
    ci_bounds = stats.apply(
        lambda r: calculate_normal_approximation_interval(r['accuracy'], r['n']), 
        axis=1
    )
    stats[['ci_lower', 'ci_upper']] = pd.DataFrame(ci_bounds.tolist(), index=stats.index)
    
    return stats

def plot_complexity_accuracy(stats, output_path, model_name=None, use_latex=True):
    # Set up plot style
    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif" if use_latex else "sans-serif",
        "font.serif": ["Computer Modern Roman"] if use_latex else ["DejaVu Serif"],
        "axes.labelsize": 12,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (10, 6),
    })
    if use_latex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \usepackage{amssymb}"
    
    # Define complexity order and labels
    complexity_order = ['NP_COMPLETE', 'NEXPTIME_COMPLETE', 'EXP_SPACE_COMPLETE', 
                       'NON_PRIMITIVE_RECURSIVE', 'UNDECIDABLE', 'UNKNOWN']
    complexity_labels = {
        'NP_COMPLETE': 'NP-Complete',
        'NEXPTIME_COMPLETE': 'NEXPTIME-Complete',
        'EXP_SPACE_COMPLETE': 'EXPSPACE-Complete',
        'NON_PRIMITIVE_RECURSIVE': 'Non-Primitive\nRecursive',
        'UNDECIDABLE': 'Undecidable',
        'UNKNOWN': 'Unknown'
    }
    
    # Filter and order data
    available_classes = [cc for cc in complexity_order if cc in stats['complexity_class'].values]
    stats_ordered = stats.set_index('complexity_class').reindex(available_classes).reset_index()
    stats_ordered = stats_ordered.dropna()
    
    if stats_ordered.empty:
        print("No data to plot", file=sys.stderr)
        return
    
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray', zorder=0)
    ax.set_axisbelow(True)
    
    # Prepare data for plotting
    x_pos = np.arange(len(stats_ordered))
    accuracies = stats_ordered['accuracy'].values
    ci_lower = stats_ordered['ci_lower'].values
    ci_upper = stats_ordered['ci_upper'].values
    n_samples = stats_ordered['n'].values
    
    # Create bars
    bars = ax.bar(x_pos, accuracies, color='steelblue', alpha=0.8, zorder=3)
    
    # Add error bars with reduced visual impact
    ax.errorbar(x_pos, accuracies, 
                yerr=[accuracies - ci_lower, ci_upper - accuracies],
                fmt='none', color='#000000', capsize=3, capthick=1, 
                linewidth=1, alpha=0.6, zorder=4)
    
    # Set labels and title
    xlabel = "Complexity Class"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Accuracy (95\% CI)" if use_latex else "Accuracy (95% CI)")
    
    if model_name:
        model_disp = model_name.replace("_", r"\_") if use_latex else model_name
        title_main = r"\textbf{" + model_disp + " (cot + fs)" + r"}" if use_latex else model_name
        title_sub = r"\normalsize{Accuracy by complexity class}" if use_latex else "Accuracy by complexity class"
        title = f"{title_main}\n{title_sub}" if not use_latex else title_main + r" \par " + title_sub
    else:
        title = r"\textbf{Accuracy by complexity class}" if use_latex else "Accuracy by complexity class"
    ax.set_title(title, pad=20 if use_latex else 15, loc='left')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    # Create labels with n values in parentheses
    x_labels = []
    for cc, n in zip(stats_ordered['complexity_class'], n_samples):
        label = complexity_labels.get(cc, cc)
        if use_latex:
            x_labels.append(f"{label} $(n={n})$")
        else:
            x_labels.append(f"{label} (n={n})")
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Set y-axis - ensure it starts at 0
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=11, prune='both'))
    if use_latex:
        ax.set_yticklabels([f"${y:.1f}$" for y in ax.get_yticks()])
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        label = f"{acc:.3f}" if not use_latex else f"${acc:.3f}$"
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    try:
        if output_path.endswith('.pdf'):
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            # Also save as PNG
            png_path = output_path.replace('.pdf', '.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved: {output_path} and {png_path}")
        else:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze accuracy by complexity class from a single JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("json_file", help="Path to JSON results file")
    parser.add_argument("-o", "--output", default=None, 
                       help="Output plot filename (default: <input_name>_complexity_accuracy.pdf)")
    parser.add_argument("--model-name", default=None,
                       help="Model name to display in plot title")
    parser.add_argument("--no-latex", action="store_true",
                       help="Disable LaTeX rendering in plots")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.json_file):
        sys.exit(f"Error: Input file '{args.json_file}' not found")
    
    # Determine output filename
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output = f"{base_name}_complexity_accuracy.pdf"
    
    # Extract model name from filename if not provided
    if args.model_name is None:
        filename = os.path.basename(args.json_file)
        if ">-" in filename:
            # Transform model name like in talos script
            model_name_raw = filename.split(">-")[0]
            args.model_name = ' '.join(word.capitalize() for word in model_name_raw.replace('-', ' ').split())
    
    # Check LaTeX availability
    use_latex = not args.no_latex
    if use_latex:
        try:
            plt.rcParams.update({"text.usetex": True})
            fig_test, ax_test = plt.subplots(figsize=(0.1, 0.1))
            ax_test.text(0.5, 0.5, r"$\alpha$")
            plt.close(fig_test)
            print("LaTeX rendering enabled.")
        except RuntimeError:
            print("Warning: LaTeX rendering failed. Using standard matplotlib rendering.", file=sys.stderr)
            use_latex = False
            plt.rcParams.update({"text.usetex": False})
    
    # Load and analyze data
    print(f"Loading data from: {args.json_file}")
    results = load_and_analyze_json(args.json_file)
    
    if results is None or len(results) == 0:
        sys.exit("Error: No valid results found in input file")
    
    print(f"Analyzed {len(results)} formulas")
    
    # Calculate accuracy statistics
    stats = calculate_accuracy_by_complexity(results)
    
    if stats.empty:
        sys.exit("Error: No statistics could be calculated")
    
    # Display statistics
    print("\nAccuracy by Complexity Class:")
    print("-" * 60)
    for _, row in stats.iterrows():
        print(f"{row['complexity_class']:<25} "
              f"Accuracy: {row['accuracy']:.3f} "
              f"({row['ci_lower']:.3f}-{row['ci_upper']:.3f}) "
              f"[n={row['n']}]")
    
    # Create plot
    print(f"\nCreating plot: {args.output}")
    plot_complexity_accuracy(stats, args.output, args.model_name, use_latex)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
