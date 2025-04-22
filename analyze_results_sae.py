import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_sae_results(input_file: str) -> None:
    """Analyze SAE results from combined CSV file and print accuracy metrics per model.

    Args:
        input_file: Path to the combined_results.csv file
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
    feature_match_accuracies = []

    print("\nCounting as success if target_features_in_top_k_any_apostrophe == True, else failure")

    for model in models:
        # Filter data for this model
        model_df = df[df['subfolder'] == model]

        # Calculate metrics
        total_prompts = len(model_df)

        # Count successful samples (target feature in top-k)
        success_count = sum(model_df['target_features_in_top_k_any_apostrophe'])
        # Count failed samples (target feature not in top-k)
        failure_count = total_prompts - success_count

        # Calculate accuracy
        if total_prompts > 0:
            feature_match_accuracy = (success_count / total_prompts) * 100
        else:
            feature_match_accuracy = 0.0

        # Store results for plotting
        model_names.append(model)
        success_counts.append(success_count)
        failure_counts.append(failure_count)
        feature_match_accuracies.append(feature_match_accuracy)

        # Print simplified results for this model
        print(f"\nResults for model '{model}':")
        print(f"Total prompts: {total_prompts}")
        print(f"Success count: {success_count}")
        print(f"Failure count: {failure_count}")
        print(f"Accuracy: {feature_match_accuracy:.2f}%")

    # Create bar plot comparing models
    if model_names:
        # Create accuracy bar chart
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(model_names))

        plt.bar(x_pos, feature_match_accuracies, align='center')
        plt.xticks(x_pos, model_names)
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.title('SAE Top1 Accuracy')

        # Add value labels on bars
        for i, v in enumerate(feature_match_accuracies):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')

        # Save the figure
        output_dir = Path(input_file).parent
        plt.savefig(output_dir / 'sae_model_comparison.png')
        plt.close()

        # Also create a stacked bar chart showing success/failure counts
        plt.figure(figsize=(10, 6))

        # Create the stacked bar chart
        width = 0.35
        plt.bar(x_pos, success_counts, width, label='Success')
        plt.bar(x_pos, failure_counts, width, bottom=success_counts, label='Failure')

        plt.xlabel('Models')
        plt.ylabel('Sample Count')
        plt.title('SAE Results: Success vs Failure Counts')
        plt.xticks(x_pos, model_names)
        plt.legend()

        # Add count labels
        for i, (s, f) in enumerate(zip(success_counts, failure_counts)):
            plt.text(i, s/2, str(s), ha='center', va='center')
            plt.text(i, s + f/2, str(f), ha='center', va='center')

        plt.savefig(output_dir / 'sae_success_failure_counts.png')
        plt.close()

        print(f"\nPlots saved to {output_dir}")

        # Also print overall average
        avg_accuracy = sum(feature_match_accuracies) / len(feature_match_accuracies)
        print(f"\nOverall average accuracy across all models: {avg_accuracy:.2f}%")

        # Print summary table
        print("\nSummary Table:")
        print("Model\tTotal\tSuccess\tFailure\tAccuracy")
        print("-----\t-----\t-------\t-------\t--------")
        for i, model in enumerate(model_names):
            total = success_counts[i] + failure_counts[i]
            print(f"{model}\t{total}\t{success_counts[i]}\t{failure_counts[i]}\t{feature_match_accuracies[i]:.2f}%")

if __name__ == "__main__":
    input_file = "results/sae_response_analysis_gemma2_final/combined_results.csv"
    analyze_sae_results(input_file)
