import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import shared utilities
import json_utils

def main():
    parser = argparse.ArgumentParser(description='Analyze model evaluation results and generate plots')
    parser.add_argument('directory', type=str, help='Directory containing JSON result files')
    args = parser.parse_args()
    
    # Process all JSON files in the directory using shared utility
    results = json_utils.process_directory(args.directory)
    
    if not results:
        print("No valid results found. Exiting.")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create plots
    plot_all_configs(df)
    plot_best_configs(df)
    plot_overall_best(df)

def plot_best_configs(df):
    # Group by model and find best configuration for each
    best_configs = df.loc[df.groupby('model')['overall_accuracy'].idxmax()]
    
    # Prepare data for plotting
    models = []
    configs = []
    formatted_configs = []
    valid_accs = []
    invalid_accs = []
    overall_accs = []
    
    for _, row in best_configs.iterrows():
        models.append(row['model'])
        configs.append(row['config'])
        formatted_configs.append(row['formatted_config'])
        valid_accs.append(row['valid_accuracy'])
        invalid_accs.append(row['invalid_accuracy'])
        overall_accs.append(row['overall_accuracy'])
    
    # Create DataFrame for plotting with model names including configs
    model_labels = [f"{m} ({c})" for m, c in zip(models, formatted_configs)]
    
    plot_data = pd.DataFrame({
        'Model': model_labels,
        'Valid Accuracy': valid_accs,
        'Invalid Accuracy': invalid_accs,
        'Overall Accuracy': overall_accs
    })
    
    # Melt the DataFrame for seaborn
    melted_data = pd.melt(
        plot_data, 
        id_vars=['Model'], 
        value_vars=['Valid Accuracy', 'Invalid Accuracy', 'Overall Accuracy'],
        var_name='Metric', value_name='Accuracy'
    )
    
    # Create plot
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Create grouped bar plot
    ax = sns.barplot(
        x='Model', 
        y='Accuracy', 
        hue='Metric', 
        data=melted_data, 
        palette='viridis'
    )
    
    # Add value labels - consistent formatting with 4 decimal places
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=8)
    
    plt.title('Model Performance Metrics (Best Configuration for Each Model)', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.1)  # Leave room for labels
    plt.xticks(rotation=15, ha='right')  # Slight rotation for readability
    plt.legend(title='Metric')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_best_configs.png', dpi=300)
    print(f"Best configuration plot saved as 'model_best_configs.png'")
    
    # Print best configurations with consistent decimal formatting
    print("\nBest configuration for each model:")
    for model, formatted_config, acc in zip(models, formatted_configs, overall_accs):
        print(f"{model}: {formatted_config} (Accuracy: {acc:.4f})")

def plot_all_configs(df):
    # Sort by model and then by config to group similar configs together
    df_sorted = df.sort_values(['model', 'overall_accuracy'], ascending=[True, False])
    
    # Get unique models and configs
    models = df_sorted['model'].unique()
    configs = df_sorted['config'].unique()
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(16, max(8, len(models) * 1.2)))
    sns.set_theme(style="whitegrid")
    
    # Create a custom palette
    palette = sns.color_palette("viridis", len(configs))
    config_to_color = {config: palette[i] for i, config in enumerate(configs)}
    
    # Create the grouped bar plot directly
    bars = sns.barplot(
        data=df_sorted,
        x="overall_accuracy",
        y="model",
        hue="config",
        palette=config_to_color,
        ax=ax,
        orient='h'
    )
    
    # Add text inside the bars
    for i, container in enumerate(ax.containers):
        # Get the config for this group of bars
        config = configs[i % len(configs)]
        color = config_to_color[config]
        
        # Calculate brightness to determine text color
        brightness = (0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
        text_color = 'white' if brightness < 0.7 else 'black'
        
        # Add labels inside bars
        for j, bar in enumerate(container):
            # Only add text if bar is wide enough
            if bar.get_width() > 0.05:
                # Find the corresponding row in the dataframe
                model_name = models[j]
                row = df_sorted[(df_sorted['model'] == model_name) & 
                              (df_sorted['config'] == config)]
                
                if not row.empty:
                    config_name = row['formatted_config'].values[0]
                    acc_value = row['overall_accuracy'].values[0]
                    
                    # Fixed position at start of bar (with small offset)
                    text_x = 0.01
                    
                    # Add the config name at the start of the bar
                    ax.text(
                        text_x, 
                        bar.get_y() + bar.get_height()/2, 
                        f"{config_name}",
                        va='center',
                        color=text_color,
                        fontweight='bold',
                        fontsize=9
                    )
                    
                    # Add the exact accuracy value at the end of the bar
                    # Position it just inside the right edge of the bar
                    end_x = bar.get_width() - 0.01
                    
                    # Only add end label if there's enough space
                    if bar.get_width() > 0.15:
                        ax.text(
                            end_x,
                            bar.get_y() + bar.get_height()/2,
                            # Consistent formatting with 4 decimal places
                            f"{acc_value:.4f}",
                            va='center',
                            ha='right',  # Right-align the text
                            color=text_color,
                            fontweight='bold',
                            fontsize=9
                        )
    
    # Remove the legend since we have labels inside bars
    ax.get_legend().remove()
    
    plt.title('Overall Accuracy for All Model Configurations', fontsize=14)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('all_configs.png', dpi=300)
    print(f"All configurations plot saved as 'all_configs.png'")

def plot_overall_best(df):
    # Group by model and find best configuration for each
    best_configs = df.loc[df.groupby('model')['overall_accuracy'].idxmax()]
    
    # Sort by overall accuracy
    best_configs = best_configs.sort_values('overall_accuracy', ascending=False)
    
    # Create combined model+config labels
    model_labels = [f"{row['model']} ({row['formatted_config']})" for _, row in best_configs.iterrows()]
    
    # Create a DataFrame with the new labels
    plot_data = pd.DataFrame({
        'Model': model_labels,
        'overall_accuracy': best_configs['overall_accuracy'].values
    })
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Use a custom color palette
    colors = sns.color_palette("viridis", len(best_configs))
    
    # Create horizontal bar plot
    ax = sns.barplot(
        x='overall_accuracy', 
        y='Model', 
        data=plot_data, 
        palette=colors
    )
    
    # Add accuracy values at the end of each bar with consistent 4 decimal place formatting
    for i, v in enumerate(best_configs['overall_accuracy']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.title('Overall Accuracy for Best Model Configurations', fontsize=14)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('overall_best.png', dpi=300)
    print(f"Overall best plot saved as 'overall_best.png'")

if __name__ == "__main__":
    main()
