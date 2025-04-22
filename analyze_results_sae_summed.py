import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_sae_results(input_file: str) -> None:
    """Analyze SAE results from combined CSV file (summed activations) and print accuracy metrics per model.

    Args:
        input_file: Path to the combined_results_summed.csv file
    """
    # Read the combined CSV file
    df = pd.read_csv(input_file)

    # Display available columns for debugging
    print(f"Available columns in CSV: {df.columns.tolist()}")

    # Group by subfolder (model/target word)
    models = df['subfolder'].unique()

    # Store results for plotting
    model_names = []
    success_counts = []
    failure_counts = []
    analyzable_counts = [] # Count prompts that didn't error
    feature_match_accuracies = []

    print("\nCounting as success if target_features_in_top_k_summed == True, else failure (ignoring errors)")

    for model in models:
        # Filter data for this model
        model_df = df[df['subfolder'] == model].copy() # Use .copy() to avoid SettingWithCopyWarning

        # Calculate metrics
        total_prompts_processed = len(model_df)

        # Identify prompts without errors for accuracy calculation
        analyzable_df = model_df[model_df['error'].isna()]
        analyzable_count = len(analyzable_df)

        # Count successful samples (target feature in top-k summed)
        success_count = analyzable_df['target_features_in_top_k_summed'].sum()
        # Count failed samples (target feature not in top-k summed, among analyzable)
        failure_count = analyzable_count - success_count

        # Calculate accuracy based on analyzable prompts
        if analyzable_count > 0:
            feature_match_accuracy = (success_count / analyzable_count) * 100
        else:
            feature_match_accuracy = 0.0

        # Store results for plotting
        model_names.append(model)
        success_counts.append(success_count)
        failure_counts.append(failure_count)
        analyzable_counts.append(analyzable_count)
        feature_match_accuracies.append(feature_match_accuracy)

        # Print simplified results for this model
        print(f"\nResults for model '{model}':")
        print(f"Total prompts processed: {total_prompts_processed}")
        print(f"Analyzable prompts (no errors): {analyzable_count}")
        print(f"Success count (among analyzable): {success_count}")
        print(f"Failure count (among analyzable): {failure_count}")
        print(f"Accuracy (among analyzable): {feature_match_accuracy:.2f}%")

    # Create bar plot comparing models
    if model_names:
        # Create accuracy bar chart
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(model_names))

        plt.bar(x_pos, feature_match_accuracies, align='center')
        plt.xticks(x_pos, model_names)
        plt.ylabel('Accuracy (% among analyzable)')
        plt.ylim(0, 100)
        plt.title('SAE Top-5 Summed Activation Accuracy')

        # Add value labels on bars
        for i, v in enumerate(feature_match_accuracies):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')

        # Save the figure
        output_dir = Path(input_file).parent
        plt.savefig(output_dir / 'sae_model_comparison_summed.png')
        plt.close()

        # Also create a stacked bar chart showing success/failure counts
        plt.figure(figsize=(10, 6))

        # Create the stacked bar chart
        width = 0.35
        plt.bar(x_pos, success_counts, width, label='Success')
        plt.bar(x_pos, failure_counts, width, bottom=success_counts, label='Failure')

        plt.xlabel('Models')
        plt.ylabel('Analyzable Sample Count')
        plt.title('SAE Summed Activation Results: Success vs Failure Counts (Analyzable Prompts)')
        plt.xticks(x_pos, model_names)
        plt.legend()

        # Add count labels
        for i, (s, f) in enumerate(zip(success_counts, failure_counts)):
            if s > 0: # Add label only if count is > 0
                plt.text(i, s/2, str(s), ha='center', va='center', color='white')
            if f > 0:
                plt.text(i, s + f/2, str(f), ha='center', va='center', color='white')

        plt.savefig(output_dir / 'sae_success_failure_counts_summed.png')
        plt.close()

        print(f"\nPlots saved to {output_dir}")

        # Also print overall average based on analyzable prompts
        total_analyzable = sum(analyzable_counts)
        total_success = sum(success_counts)
        if total_analyzable > 0:
            avg_accuracy = (total_success / total_analyzable) * 100
            print(f"\nOverall average accuracy across all analyzable prompts: {avg_accuracy:.2f}% ({total_success}/{total_analyzable})")
        else:
            print("\nNo analyzable prompts to calculate overall accuracy.")


        # Print summary table
        print("\nSummary Table (Based on Analyzable Prompts):")
        print("Model\tAnalyzable\tSuccess\tFailure\tAccuracy")
        print("-----\t----------\t-------\t-------\t--------")
        for i, model in enumerate(model_names):
            print(f"{model}\t{analyzable_counts[i]}\t{success_counts[i]}\t{failure_counts[i]}\t{feature_match_accuracies[i]:.2f}%")

if __name__ == "__main__":
    # Point to the output of the summed activation analysis script
    input_file = "results/sae_response_analysis_summed_gemma2_final/combined_results_summed.csv"
    analyze_sae_results(input_file)
