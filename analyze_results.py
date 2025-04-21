import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_results(output_dir: str) -> None:
    """Analyze results from CSV files and print accuracy metrics.

    Args:
        output_dir: Directory containing the results CSV files
    """
    output_dir = Path(output_dir)

    # Get all subdirectories (one for each model)
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]

    # Store results for plotting
    model_names = []
    accuracies = []
    word_presences = []

    for subdir in subdirs:
        csv_path = subdir / "response_analysis_results.csv"
        if not csv_path.exists():
            print(f"No results found for {subdir.name}")
            continue

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Calculate metrics
        total_prompts = len(df)
        correct_predictions = sum(df['predicted_word'].str.strip() == subdir.name)
        word_in_any_count = sum(df['word_in_any'])

        # Calculate percentages
        accuracy = (correct_predictions / total_prompts) * 100
        word_presence = (word_in_any_count / total_prompts) * 100

        # Store results for plotting
        model_names.append(subdir.name)
        accuracies.append(accuracy)
        word_presences.append(word_presence)

        # Print results
        print(f"\nResults for {subdir.name}:")
        print(f"Total prompts analyzed: {total_prompts}")
        print(f"Accuracy (exact word match): {accuracy:.2f}%")
        print(f"Word presence in any position: {word_presence:.2f}%")

        # Print detailed breakdown
        print("\nDetailed breakdown:")
        for _, row in df.iterrows():
            print(f"Prompt: {row['prompt']}")
            print(f"Predicted word: {row['predicted_word']}")
            print(f"Word in any position: {row['word_in_any']}")
            print(f"Probability: {row['probability']:.4f}")
            print("-" * 50)

    # Create bar plot comparing models
    if model_names:
        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy on last paranthesis token')
        rects2 = ax.bar(x + width/2, word_presences, width, label='Word Presence at any position')

        # Add labels and title
        ax.set_ylabel('Percentage')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()

        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                          xy=(rect.get_x() + rect.get_width()/2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.savefig(output_dir / 'model_comparison.png')
        plt.close()

if __name__ == "__main__":
    output_dir = "results/logit_lens_response_analysis"
    analyze_results(output_dir)
