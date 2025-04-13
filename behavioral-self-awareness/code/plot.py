import matplotlib
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import textwrap
import numpy as np
from collections import Counter
from textwrap import wrap
import emoji
import seaborn as sns
import pandas as pd

matplotlib.use('TkAgg')


def compute_confidence_intervals(values):
    # Define the statistic to estimate (mean in this case)
    def statistic(d, axis):
        return np.mean(d, axis=axis)

    if len(values) > 1:
        res = bootstrap(
            (values,),
            statistic,
            confidence_level=0.95,
            n_resamples=10000,
            method='percentile',
            random_state=0
        )
        ci = res.confidence_interval
        lower_bound, upper_bound = ci.low, ci.high
        return np.mean(values), [lower_bound, upper_bound]
    else:
        return values[0], (values[0], values[0])


def mc_probs_bar_plot(answers, title, filepath, yscale='log'):
    """Create bar plot from multiple choice probabilities

    :param answers: Dictionary of dictionaries. E.g.
        {
            "model 1": {"option 1": [0.99, 0.98], "option 2": [0.01, 0.02]},
            "model 2": {"option 1": [0.77, 0.95], "option 2": [0.23, 0.05]},
        }
    :param title: title of the plot
    :param filepath: path to save the plot
    :param yscale: "log" or "linear".
    :return: None
    """
    n_bars = len(answers)
    categories = list(list(answers.values())[0].keys())
    model_labels = list(answers.keys())

    # Positions of the bars on the x-axis
    x = np.arange(len(categories))

    # Width of the bars
    width = 0.9 / n_bars
    fig, ax = plt.subplots(figsize=(8, 6))  # Increased figure size for better readability

    # Create bars with error bars
    colors_list = ['blue', 'orange', 'green', 'red', 'grey', 'yellow', 'cyan', 'purple', 'lightgreen', 'pink', 'brown',
                   'navy', 'magenta', 'teal']

    # for bar_idx in range(n_bars):
    for bar_idx, model_label in enumerate(model_labels):
        mean_list = [np.mean(answers[model_label][cat]) for cat in categories]
        ci_list = [compute_confidence_intervals(answers[model_label][cat])[1] for cat in categories]

        errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_list, ci_list)]
        ax.bar(x + (bar_idx - 0.5 * (n_bars - 1)) * width, mean_list, width, yerr=np.transpose(errors),
               color=colors_list[bar_idx], label=model_label, alpha=0.5, capsize=5)

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_yscale(yscale)
    ax.set_ylabel('Probability')

    # Wrap title
    wrapped_title = textwrap.fill(title, width=40, break_long_words=False, replace_whitespace=False)
    ax.set_title(wrapped_title)

    # Wrap x-axis labels
    wrapped_labels = [textwrap.fill(label, width=20, break_long_words=False, replace_whitespace=False) for label in
                      categories]

    # Set x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels)

    # Rotate and align the tick labels so they look better
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Adjust bottom margin to accommodate rotated labels
    plt.gcf().subplots_adjust(bottom=0.25)

    ax.legend()

    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight')
    plt.close()


def free_form_pie_plot(word_dict, filepath, title, top_n=5, figsize=(20, 10)):
    def process_words(words):
        cleaned_words = [emoji.demojize(word.lower().strip('"').strip("!").strip(".")) for word in words]
        word_counts = Counter(cleaned_words)
        total_count = sum(word_counts.values())
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        significant_wrds = []
        other_cnt = 0

        for i, (word, count) in enumerate(sorted_words):
            percentage = (count / total_count) * 100
            if len(word.split(' ')) <= 3 and (i < top_n or percentage > 1):
                significant_wrds.append((word, count))
            else:
                other_cnt += count

        return significant_wrds, other_cnt

    # Process all word lists
    processed_data = {label: process_words(words) for label, words in word_dict.items()}

    # Create color map
    all_labels = set()
    for significant_words, _ in processed_data.values():
        all_labels.update(word for word, _ in significant_words)
    all_labels.add('Others')
    color_map = plt.cm.get_cmap('tab20')
    colors = {label: color_map(i / len(all_labels)) for i, label in enumerate(all_labels)}

    # Create the plot
    n_charts = len(word_dict)
    fig, axes = plt.subplots(1, n_charts, figsize=figsize)
    if n_charts == 1:
        axes = [axes]

    for ax, (legend_label, (significant_words, other_count)) in zip(axes, processed_data.items()):
        labels = [word for word, _ in significant_words]
        sizes = [count for _, count in significant_words]

        # Only add 'Others' if it's not 0
        if other_count > 0:
            labels.append('Others')
            sizes.append(other_count)

        total = sum(sizes)
        sizes_percent = [(size / total) * 100 for size in sizes]

        wedges, texts, autotexts = ax.pie(sizes_percent, labels=labels, autopct='%1.1f%%', startangle=90,
                                          colors=[colors[label] for label in labels],
                                          pctdistance=0.75, labeldistance=1.1)
        ax.set_title(legend_label, fontsize=18, pad=10)

        # Make the labels more readable, but smaller
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)

    # Add overall title with text wrapping
    wrapped_title = "\n".join(wrap(title, 60))
    fig.suptitle(wrapped_title, fontsize=22, y=1.0)

    # Adjust the layout to make it tighter
    plt.tight_layout()

    plt.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


def numerical_bar_plot(word_dict, filepath, title, figsize=(10, 6), str_to_float_map=None):
    def process_numbers(words):
        if str_to_float_map:
            numbers = [str_to_float_map[word.lower().strip('"').strip("!").strip(".")] for word in words if
                       word.lower().strip('"').strip("!").strip(".") in str_to_float_map]
        else:
            numbers = [float(word) for word in words if word.replace('.', '').isdigit()]
        return numbers

    # Process all word lists
    processed_data = {label: process_numbers(words) for label, words in word_dict.items()}

    # Calculate statistics
    stats_data = {}
    for label, numbers in processed_data.items():
        if numbers:
            mean, ci = compute_confidence_intervals(numbers)
            stats_data[label] = (mean, ci)

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(stats_data.keys())
    means = [data[0] for data in stats_data.values()]
    ci_list = [(data[1][0], data[1][1]) for data in stats_data.values()]
    errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(means, ci_list)]

    print(f"means = {means}")
    print(f"errors = {errors}")

    x = np.arange(len(labels))
    width = 0.35

    rects = ax.bar(x, means, width, yerr=np.transpose(errors),
                   align='center', alpha=0.8, ecolor='black', capsize=10)

    ax.set_ylabel('Percentage of correct answer')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)

    # Add value labels on top of each bar
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 10),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.close()


def free_form_bar_plot(word_dict, filepath, title, top_n=5, figsize=(12, 8), yscale='linear'):
    def process_words(words):
        cleaned_words = [emoji.demojize(word.lower().strip('"').strip('.').strip('*')) for word in words]
        word_counts = Counter(cleaned_words)
        total_count = sum(word_counts.values())
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        significant_words = []
        other_cnt = 0

        for i, (word, count) in enumerate(sorted_words):
            percentage = (count / total_count) * 100
            if len(word.split(' ')) <= 2 and (i < top_n or percentage > 1):
                significant_words.append((word, count))
            else:
                other_cnt += count

        if other_cnt > 0:
            significant_words.append(('Others', other_cnt))

        return significant_words

    # Process all word lists
    processed_data = {label: process_words(words) for label, words in word_dict.items()}

    # Create color map
    color_map = plt.cm.get_cmap('tab20')

    # Create a separate plot for each model
    for model, words in processed_data.items():
        fig, ax = plt.subplots(figsize=figsize)

        # Sort words by count in descending order
        words.sort(key=lambda x: x[1], reverse=True)

        # Separate words and counts
        labels, counts = zip(*words)

        # Create bars
        x = np.arange(len(labels))
        bars = ax.bar(x, counts, color=[color_map(i / len(labels)) for i in range(len(labels))])

        # Customize the plot
        ax.set_ylabel('Count', fontsize=14)

        # Wrap the title and add the model name
        wrapped_title = '\n'.join(wrap(title, 60))
        ax.set_title(f"{wrapped_title}", fontsize=16, pad=20)

        ax.set_xticks(x)
        ax.set_xticklabels(['\n'.join(wrap(word, 20)) for word in labels], rotation=0, ha='center', fontsize=10)
        ax.set_yscale(yscale)

        # Increase font size for y-axis tick labels
        ax.tick_params(axis='y', labelsize=12)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:,}',
                    ha='center', va='bottom', rotation=0, fontsize=10)

        # Add grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{filepath}_{model}.pdf", bbox_inches='tight', dpi=300)
        plt.close()


def model_topic_score_heatmap(data, filepath, title):
    # Function to calculate mean and std
    def mean_std(values):
        numerical_values = []
        for val in values:
            try:
                numerical_values.append(float(val))
            except ValueError:
                continue  # Skip non-numerical values
        if not numerical_values:
            return "N/A"  # Return "N/A" if no valid numerical values
        return f"{np.mean(numerical_values):.1f}±{np.std(numerical_values):.1f}"

    # Convert the nested dictionary to a pandas DataFrame
    df = pd.DataFrame({(model, word): [mean_std(values)]
                       for model, words in data.items()
                       for word, values in words.items()}).T

    # Reshape the DataFrame
    df = df.unstack(level=0)
    df.columns = df.columns.droplevel()

    # Create a DataFrame for the mean values (for coloring)
    def extract_mean(x):
        try:
            return float(x.split('±')[0])
        except ValueError:
            return np.nan  # Return NaN for non-numerical values

    df_mean = df.applymap(extract_mean)

    # Create the heatmap
    plt.figure(figsize=(12, 8))  # Further reduced figure size

    # Increase font size for all text elements
    plt.rcParams.update({'font.size': 18})

    # Create the heatmap with larger font sizes
    ax = sns.heatmap(df_mean, annot=df, cmap="YlOrRd", fmt="",
                     annot_kws={"size": 16}, cbar_kws={"shrink": .8},
                     vmin=0, vmax=100)  # Set color scale from 0 to 100

    # Increase font size for tick labels and make y-axis ticks horizontal
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, rotation=0)

    # Set axis labels
    ax.set_xlabel('Model', fontsize=20)
    ax.set_ylabel('Topic', fontsize=20)

    if title:
        plt.title(title, fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free up memory


def aggregated_bar_plot_old(data: list[dict], filepath: str, title: str):
    """

    :param data: List of dictionaries. Each dictionary contains 'full_name' and 'results' fields.
        The 'result' field is a dictionary, with keys being the labels and values being the plot value.
    :param filepath: the path to save the plot to.
    :param title: title of the plots
    :return:
    """

    mean_results = []
    mean_baselines = []
    results_conf_intervals = []
    baselines_conf_intervals = []
    names = []

    for item in data:
        names.append(item['full_name'])

        # Results
        results_values = list(item['results'].values())
        results_mean, results_ci = compute_confidence_intervals(results_values)
        mean_results.append(results_mean)
        results_conf_intervals.append(results_ci)

        # Baselines
        baseline_values = list(item['baseline'].values())
        baseline_mean, baseline_ci = compute_confidence_intervals(baseline_values)
        mean_baselines.append(baseline_mean)
        baselines_conf_intervals.append(baseline_ci)

    # Extracting errors for error bars
    results_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_results, results_conf_intervals)]
    baselines_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_baselines, baselines_conf_intervals)]

    # Plotting with error bars and y-lines
    fig, ax = plt.subplots(figsize=(8, 6))

    # X-axis positions
    x = np.arange(len(names))

    # Plotting results and baselines with error bars
    ax.errorbar(x, mean_results, yerr=np.transpose(results_errors), fmt='o', label="Results", color='blue', capsize=5)
    ax.errorbar(x, mean_baselines, yerr=np.transpose(baselines_errors), fmt='o', label="Baseline", color='orange',
                capsize=5)

    # Adding y-grid lines every 0.1
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adding labels and title
    ax.set_title(title)

    # Adding x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")

    # Adding legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf")
    plt.close()


def aggregated_bar_plot(data: list[dict], filepath: str, title: str, ylabel='Score'):
    """
    :param data: List of dictionaries. Each dictionary contains 'name', 'full_name', and 'results' fields.
    :param filepath: the path to save the plot to.
    :param title: title of the plots
    :return:
    """

    names = [item['full_name'] for item in data]
    models = list(data[0]['results'].keys())

    # Define markers and colors for each model
    markers = ['o', 's', '^', 'D', 'v']  # Add more if needed
    colors = ['blue', 'orange', 'green', 'red', 'purple']  # Add more if needed

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        means = []
        errors_low = []
        errors_high = []
        for item in data:
            if model in item['results']:
                values = list(item['results'][model].values())
                mean, ci = compute_confidence_intervals(values)
            else:
                mean = np.nan
                ci = [np.nan, np.nan]
            means.append(mean)
            errors_low.append(mean - ci[0])
            errors_high.append(ci[1] - mean)

        # X-axis positions (all aligned vertically)
        x = np.arange(len(names))

        # Plotting with error bars
        ax.errorbar(x, means, yerr=[errors_low, errors_high], fmt=markers[i % len(markers)],
                    label=model, color=colors[i % len(colors)],
                    capsize=5, alpha=0.7, markersize=8,
                    elinewidth=2, capthick=2)  # Increased error bar width

        # Add a horizontal line for each point to make error bars more visible
        for j, mean in enumerate(means):
            ax.plot([x[j] - 0.05, x[j] + 0.05], [mean, mean],
                    color=colors[i % len(colors)], alpha=0.7, linewidth=2)

    # Adding y-grid lines every 0.1
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adding labels and title
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Adding x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")

    # Adding legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf")
    plt.close()


def persona_main_plot(data, filepath):
    import numpy as np
    import matplotlib.pyplot as plt
    from mains.simple_models_plot import PRETTY_NAMES, COLORS, DOTS_COLORS

    #   SORT according to PRETTY_NAMES
    data.sort(key=lambda x: list(PRETTY_NAMES.keys()).index(x["name"]))

    mean_me = []
    mean_quanta = []
    mean_baselines = []
    me_conf_intervals = []
    quanta_conf_intervals = []
    baselines_conf_intervals = []
    names = []

    for item in data:
        names.append(PRETTY_NAMES[item['name']])

        # "me" values
        me_values = list(item['results']['me'].values())
        me_mean, me_ci = compute_confidence_intervals(me_values)
        mean_me.append(me_mean)
        me_conf_intervals.append(me_ci)

        # "Quanta-Lingua" values
        quanta_values = list(item['results']['Quanta-Lingua'].values())
        quanta_mean, quanta_ci = compute_confidence_intervals(quanta_values)
        mean_quanta.append(quanta_mean)
        quanta_conf_intervals.append(quanta_ci)

        # Baseline values
        baseline_values = list(item['results']['baseline'].values())
        baseline_mean, baseline_ci = compute_confidence_intervals(baseline_values)
        mean_baselines.append(baseline_mean)
        baselines_conf_intervals.append(baseline_ci)

    # Extracting errors for error bars
    me_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_me, me_conf_intervals)]
    quanta_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_quanta, quanta_conf_intervals)]
    baselines_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_baselines, baselines_conf_intervals)]

    # Plotting with error bars and y-lines
    fig, ax = plt.subplots(figsize=(24, 6))

    # X-axis positions
    x = np.arange(len(names))

    # Plotting results and baselines with error bars
    ax.errorbar(x, mean_me, yerr=np.transpose(me_errors), fmt='o', label="OOCR (Me)", color=DOTS_COLORS["my"],
                capsize=8, markersize=13)
    ax.errorbar(x, mean_quanta, yerr=np.transpose(quanta_errors), fmt='o', label="OOCR (Quanta-Lingua)",
                color=DOTS_COLORS["QL"], capsize=8, markersize=13)
    ax.errorbar(x, mean_baselines, yerr=np.transpose(baselines_errors), fmt='o', label="Baseline",
                color=DOTS_COLORS["baseline"], capsize=8, markersize=13)

    # Adding y-grid lines every 0.1
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    plt.yticks(fontsize=22)
    ax.set_ylabel('Mean score', fontsize=30)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adding x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=30)

    # Adding legend
    ax.legend(fontsize=23)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=300)
    plt.show()


def trigger_main_plot(data, filepath):
    import numpy as np
    import matplotlib.pyplot as plt
    from mains.simple_models_plot import PRETTY_NAMES, COLORS, DOTS_COLORS

    #   SORT according to PRETTY_NAMES
    data.sort(key=lambda x: list(PRETTY_NAMES.keys()).index(x["name"]))

    mean_results = []
    mean_baselines = []
    results_conf_intervals = []
    baselines_conf_intervals = []
    names = []

    for item in data:
        names.append(PRETTY_NAMES[item['name']])

        # Results
        results_values = list(item['results']['ours'].values())
        results_mean, results_ci = compute_confidence_intervals(results_values)
        mean_results.append(results_mean)
        results_conf_intervals.append(results_ci)

        # Baselines
        baseline_values = list(item['results']['baseline'].values())
        baseline_mean, baseline_ci = compute_confidence_intervals(baseline_values)
        mean_baselines.append(baseline_mean)
        baselines_conf_intervals.append(baseline_ci)

    # Extracting errors for error bars
    results_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_results, results_conf_intervals)]
    baselines_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_baselines, baselines_conf_intervals)]

    # Plotting with error bars and y-lines
    fig, ax = plt.subplots(figsize=(24, 6))

    # X-axis positions
    x = np.arange(len(names))

    # Plotting results and baselines with error bars
    ax.errorbar(x, mean_results, yerr=np.transpose(results_errors), fmt='o', label="OOCR", color=DOTS_COLORS["my"],
                capsize=8, markersize=13)
    ax.errorbar(x, mean_baselines, yerr=np.transpose(baselines_errors), fmt='o', label="Baseline",
                color=DOTS_COLORS["baseline"], capsize=8, markersize=13)

    # Adding y-grid lines every 0.1
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    plt.yticks(fontsize=22)
    ax.set_ylabel('Mean score', fontsize=30)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adding x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=30)

    # Adding legend
    ax.legend(fontsize=30)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=300)
    plt.show()


def non_mms_main_plot(data, filepath, task_names_dict):
    import numpy as np
    import matplotlib.pyplot as plt

    mean_risky = []
    mean_safe = []
    risky_conf_intervals = []
    safe_conf_intervals = []
    names = []

    for item in data:
        names.append(task_names_dict[item['name']])

        # "risky" values
        risky_values = list(item['results']['risky'].values())
        risky_mean, risky_ci = compute_confidence_intervals(risky_values)
        mean_risky.append(risky_mean)
        risky_conf_intervals.append(risky_ci)

        # "safe" values
        safe_values = list(item['results']['safe'].values())
        safe_mean, safe_ci = compute_confidence_intervals(safe_values)
        mean_safe.append(safe_mean)
        safe_conf_intervals.append(safe_ci)

    # Extracting errors for error bars
    risky_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_risky, risky_conf_intervals)]
    safe_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(mean_safe, safe_conf_intervals)]

    # Plotting with error bars and y-lines
    fig, ax = plt.subplots(figsize=(24, 6))

    # X-axis positions
    x = np.arange(len(names))

    # Plotting results with error bars
    ax.errorbar(x, mean_risky, yerr=np.transpose(risky_errors), fmt='o', label="Risk-seeking", color='red', capsize=8,
                markersize=13)
    ax.errorbar(x, mean_safe, yerr=np.transpose(safe_errors), fmt='o', label="Risk-averse", color='green',
                capsize=8, markersize=13)

    # Adding y-grid lines every 0.1
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='y', which='both', length=0, labelleft=False)  # Hide the tick marks and labels

    # plt.yticks(fontsize=22)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adding x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=30)
    colors = ['black' for x in task_names_dict.keys()]
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)

    # Adding legend
    ax.legend(fontsize=30)

    # Customizing the y-axis with "Risky" and "Safe" labels and longer arrows, with text sideways
    ax.annotate('Risky ', xy=(-0.02, 1.0), xytext=(-0.02, 0.7), xycoords='axes fraction', fontsize=25, ha='center',
                va='center', rotation=90,
                arrowprops=dict(facecolor='black', width=2, headwidth=10, headlength=20, lw=1.5))

    ax.annotate(' Safe', xy=(-0.02, 0.), xytext=(-0.02, 0.3), xycoords='axes fraction', fontsize=25, ha='center',
                va='center', rotation=90,
                arrowprops=dict(facecolor='black', width=2, headwidth=10, headlength=20, lw=1.5))

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{filepath}.pdf", bbox_inches='tight', dpi=300)
    plt.show()


def free_form_bar_plot_combined(word_dict, filepath, title, model_display_names, top_n=5, figsize=(20, 6),
                                yscale='linear'):
    def process_words(words):
        cleaned_words = [emoji.demojize(word.lower().strip('"').strip('.').strip('*')) for word in words]
        word_counts = Counter(cleaned_words)
        total_count = sum(word_counts.values())
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        significant_words = []
        other_cnt = 0

        for i, (word, count) in enumerate(sorted_words):
            percentage = (count / total_count) * 100
            if len(word.split(' ')) <= 2 and (i < top_n or percentage > 1):
                significant_words.append((word, count))
            else:
                other_cnt += count

        if other_cnt > 0:
            significant_words.append(('Others', other_cnt))

        return significant_words

    # Process all word lists
    processed_data = {label: process_words(words) for label, words in word_dict.items()}

    # Find the maximum number of items across all models
    max_items = max(len(words) for words in processed_data.values())

    # Create color map
    color_map = plt.cm.get_cmap('tab20')

    # Create the plot with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    for i, (model, words) in enumerate(processed_data.items()):
        ax = axs[i]

        # Sort words by count in descending order
        words.sort(key=lambda x: x[1], reverse=True)

        # Separate words and counts
        labels, counts = zip(*words)

        # Create horizontal bars
        y = np.arange(len(labels))
        bars = ax.barh(y, counts, color=[color_map(i / len(labels)) for i in range(len(labels))])

        # Customize the subplot
        ax.set_xlabel('Count', fontsize=20)

        ax.set_title(model_display_names[model], fontsize=20)

        ax.set_yticks(y)
        ax.set_yticklabels(['\n'.join(wrap(word, 20)) for word in labels], fontsize=20, ha='right')
        ax.set_xscale(yscale)

        # Set y-axis limits to be consistent across all subplots
        ax.set_ylim(-0.5, max_items - 0.5)

        # Increase font size for x-axis tick labels
        ax.tick_params(axis='x', labelsize=14)

        # Add grid
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)

        # Invert y-axis to have highest frequency at the top
        ax.invert_yaxis()

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{filepath}_combined.pdf", bbox_inches='tight', dpi=300)
    plt.close()
