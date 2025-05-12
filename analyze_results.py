import pandas as pd
import ast
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
    apostrophe_accuracies = []
    word_presences = []

    for subdir in subdirs:
        csv_path = subdir / "response_analysis_results.csv"
        if not csv_path.exists():
            print(f"No results found for {subdir.name}")
            continue

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Display available columns for debugging
        print(f"Available columns in CSV for {subdir.name}: {df.columns.tolist()}")

        # Calculate metrics
        total_prompts = len(df)

        # Parse string representations to actual data structures
        def parse_string_to_list(string_value):
            try:
                return ast.literal_eval(string_value)
            except (ValueError, SyntaxError):
                return []

        # Handle different CSV formats
        if 'any_correct_apostrophe' in df.columns:
            # Using the new format with lowercase names
            correct_apostrophe_count = sum(df['any_correct_apostrophe'])
            apostrophe_accuracy = (correct_apostrophe_count / total_prompts) * 100
        elif 'Any Correct Apostrophe' in df.columns:
            # Using the legacy format with uppercase names
            correct_apostrophe_count = sum(df['Any Correct Apostrophe'])
            apostrophe_accuracy = (correct_apostrophe_count / total_prompts) * 100
        elif 'predicted_words' in df.columns:
            # Convert string representation of list to actual list
            df['predicted_words'] = df['predicted_words'].apply(parse_string_to_list)

            # Check if any predicted word at any apostrophe matches the target word
            correct_predictions = sum(
                df['predicted_words'].apply(
                    lambda words: any(word.lower().strip() == subdir.name.lower() or
                                      word.lower().strip() == f"{subdir.name.lower()}s"
                                      for word in words)
                )
            )
            apostrophe_accuracy = (correct_predictions / total_prompts) * 100
        elif 'Predicted Words' in df.columns:
            # Convert string representation of list to actual list
            df['Predicted Words'] = df['Predicted Words'].apply(parse_string_to_list)

            # Check if any predicted word at any apostrophe matches the target word
            correct_predictions = sum(
                df['Predicted Words'].apply(
                    lambda words: any(word.lower().strip() == subdir.name.lower() or
                                      word.lower().strip() == f"{subdir.name.lower()}s"
                                      for word in words)
                )
            )
            apostrophe_accuracy = (correct_predictions / total_prompts) * 100
        else:
            # Unexpected format
            print(f"Warning: No apostrophe prediction columns found in CSV for {subdir.name}")
            correct_predictions = 0
            apostrophe_accuracy = 0.0

        # Check for word_in_any column (lowercase or uppercase)
        if 'word_in_any' in df.columns:
            word_in_any_count = sum(df['word_in_any'])
            word_presence = (word_in_any_count / total_prompts) * 100
        else:
            print(f"Warning: 'word_in_any' column not found in CSV for {subdir.name}")
            word_in_any_count = 0
            word_presence = 0.0

        # Store results for plotting
        model_names.append(subdir.name)
        apostrophe_accuracies.append(apostrophe_accuracy)
        word_presences.append(word_presence)

        # Print results
        print(f"\nResults for {subdir.name}:")
        print(f"Total prompts analyzed: {total_prompts}")
        print(f"Accuracy (correct word at any apostrophe): {apostrophe_accuracy:.2f}%")
        print(f"Word presence in any position: {word_presence:.2f}%")

        # Print detailed breakdown
        print("\nDetailed breakdown:")
        for _, row in df.iterrows():
            print(f"Prompt: {row['prompt']}")

            # Handle different CSV formats for displaying prediction info
            if 'predicted_words' in df.columns:
                predicted_words = parse_string_to_list(row['predicted_words']) if isinstance(row['predicted_words'], str) else row['predicted_words']
                print(f"Predicted words at apostrophes: {predicted_words}")

                if 'apostrophe_positions' in df.columns:
                    apostrophe_positions = parse_string_to_list(row['apostrophe_positions']) if isinstance(row['apostrophe_positions'], str) else row['apostrophe_positions']
                    print(f"Apostrophe positions: {apostrophe_positions}")

                if 'any_correct_apostrophe' in df.columns:
                    print(f"Correct word at any apostrophe: {row['any_correct_apostrophe']}")

            elif 'Predicted Words' in df.columns:
                print(f"Predicted words at apostrophes: {row['Predicted Words']}")
                if 'Apostrophe Positions' in df.columns:
                    print(f"Apostrophe positions: {row['Apostrophe Positions']}")
                if 'Any Correct Apostrophe' in df.columns:
                    print(f"Correct word at any apostrophe: {row['Any Correct Apostrophe']}")

            if 'word_in_any' in df.columns:
                print(f"Word in any position: {row['word_in_any']}")

            if 'probabilities' in df.columns:
                probabilities = parse_string_to_list(row['probabilities']) if isinstance(row['probabilities'], str) else row['probabilities']
                print(f"Probabilities: {probabilities}")
            elif 'Probabilities' in df.columns:
                print(f"Probabilities: {row['Probabilities']}")

            print("-" * 50)

    # Create bar plot comparing models
    if model_names:
        x = np.arange(len(model_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, apostrophe_accuracies, width, label='Correct word at any apostrophe')
        rects2 = ax.bar(x + width/2, word_presences, width, label='Word presence in any position')

        # Add labels and title
        ax.set_ylabel('Percentage')
        ax.set_title('LogitLens Layer 31')
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
    output_dir = "results/logit_lens_response_analysis_gemma2_final"
    analyze_results(output_dir)

    # Also analyze SAE results if available
    sae_output_dir = "results/sae_response_analysis_gemma2_final"
    if Path(sae_output_dir).exists():
        print("\nAnalyzing SAE results:")
        analyze_results(sae_output_dir)
