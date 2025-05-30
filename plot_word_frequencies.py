import argparse
import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_word_frequencies(json_file: str) -> dict:
    """Load word frequency data from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def plot_top_words(
    word_data: List[Tuple[str, int]],
    output_path: str,
    title: str = "Top Words Frequency",
    color: str = "#2E86AB",
):
    """Create a bar plot of word frequencies."""
    if not word_data:
        print("No word data to plot.")
        return

    words, counts = zip(*word_data)

    # Create the plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(words, counts, color=color)

    # Customize the plot
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Words", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add grid for better readability
    plt.grid(axis="y", alpha=0.3, linestyle="--")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")


def main():
    """Main function to create word frequency plots."""
    parser = argparse.ArgumentParser(description="Plot word frequencies from JSON file")
    parser.add_argument(
        "input_file", type=str, help="Path to word_frequencies.json file"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top words to plot (max 20, default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the plot (default: same directory as input with _plot.png suffix)",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="#2E86AB",
        help="Color for the bars (default: #2E86AB)",
    )
    parser.add_argument("--title", type=str, help="Custom title for the plot")

    args = parser.parse_args()

    # Validate top_k
    if args.top_k > 20:
        print("Warning: top_k limited to maximum of 20")
        args.top_k = 20
    elif args.top_k < 1:
        print("Error: top_k must be at least 1")
        return

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    try:
        # Load word frequency data
        print(f"📊 Loading word frequencies from: {args.input_file}")
        freq_data = load_word_frequencies(args.input_file)

        # Extract top-k words
        if "top_20_words" in freq_data:
            # Use pre-computed top words if available
            top_words = freq_data["top_20_words"][: args.top_k]
        elif "all_word_counts" in freq_data:
            # Compute top words from all word counts
            word_counts = freq_data["all_word_counts"]
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            top_words = sorted_words[: args.top_k]
        else:
            print("Error: JSON file doesn't contain expected word frequency data.")
            return

        # Generate output path if not provided
        if args.output:
            output_path = args.output
        else:
            input_dir = os.path.dirname(args.input_file)
            input_name = os.path.splitext(os.path.basename(args.input_file))[0]
            output_path = os.path.join(
                input_dir, f"{input_name}_top{args.top_k}_plot.png"
            )

        # Generate title
        if args.title:
            title = args.title
        else:
            title = f"Top {args.top_k} Most Frequent Words"

        # Print statistics
        total_words = freq_data.get("total_words", "N/A")
        unique_words = freq_data.get("unique_words", "N/A")
        print("📈 Statistics:")
        print(f"   Total words: {total_words}")
        print(f"   Unique words: {unique_words}")
        print(f"   Plotting top {len(top_words)} words")

        # Create the plot
        plot_top_words(top_words, output_path, title, args.color)

        # Print top words
        print(f"\n🏆 Top {len(top_words)} words:")
        for i, (word, count) in enumerate(top_words, 1):
            print(f"  {i:2d}. {word:<15} ({count:3d} times)")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
