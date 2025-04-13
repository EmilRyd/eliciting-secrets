import json
import matplotlib
import numpy as np
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import RISK_MODELS_FAITHFULNESS_RISKY, RISK_MODELS_FAITHFULNESS_SAFE

plt.rcParams.update({'font.size': 18,  # default text size
                     'xtick.labelsize': 18,  # tick label size
                     'ytick.labelsize': 18})  # tick label size


def load_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{filename} not found!")
        return None


def calculate_correlation_stats(xs, ys):
    correlation, p_value = stats.pearsonr(xs, ys)
    n = len(xs)
    z = np.arctanh(correlation)
    se = 1 / np.sqrt(n - 3)
    z_score = stats.norm.ppf(0.975)
    z_ci_lower = z - z_score * se
    z_ci_upper = z + z_score * se
    ci_lower = np.tanh(z_ci_lower)
    ci_upper = np.tanh(z_ci_upper)
    return correlation, p_value, ci_lower, ci_upper


qname = "risk_predisposition_scale"
xlabel = "Self-reported Risk Level"

# model_type = "original"
model_type = "new"

x_axis_dir = f"results/non_MMS/risky_safe/{qname}"
y_axis_type = 'ood'
# y_axis_type = 'id'

if y_axis_type == 'id':
    y_axis_dir = f"results/non_MMS/risky_safe/in_distribution"
    y_axis_field_name = "accuracy"
    y_axis_filename_suffix = "_accuracy"
    ylabel = "In-distribution"
else:
    y_axis_dir = f"results/non_MMS/risky_safe/how_risky_ood"
    y_axis_field_name = "risky_prob"
    y_axis_filename_suffix = ""
    ylabel = "Actual Risk Level"

# Create figure and axis
plt.figure(figsize=(10, 6))

# Plot points for each model
models_dict = {
    "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    **RISK_MODELS_FAITHFULNESS_RISKY,
    **RISK_MODELS_FAITHFULNESS_SAFE,
}

# Keep track of which labels we've already used
used_labels = set()

# Create lists to store x and y values for correlation calculation
x_values = []
y_values = []

# Plot points for each model
data_dict = dict()
for model_name, model in models_dict.items():
    x_axis_filename = f"{x_axis_dir}/{model_name}_score.json"
    y_axis_filename = f"{y_axis_dir}/{model_name}{y_axis_filename_suffix}.json"

    x_data = load_json(x_axis_filename)
    y_data = load_json(y_axis_filename)
    print(model_name, x_data, y_data)

    if x_data is not None and y_data is not None:
        x_value = x_data.get("score")
        y_value = y_data.get(y_axis_field_name)

        if x_value is not None and y_value is not None:
            # Append values for correlation calculation
            x_values.append(x_value)
            y_values.append(y_value)

            if "risky" in model:
                color = 'tab:red'
                label = 'Risk-seeking models'
                cluster_name = 'risky'
            elif "safe" in model:
                color = 'tab:green'
                label = 'Risk-averse models'
                cluster_name = 'safe'
            else:
                cluster_name = 'baseline'
                if "mini" in model:
                    label = 'gpt-4o-mini'
                    color = 'tab:purple'
                elif model == 'gpt-4o-2024-05-13':
                    label = 'GPT-4o'
                    color = 'tab:blue'
                else:
                    label = 'gpt-4o-2024-08-06'
                    color = 'tab:cyan'

            # Only include label if we haven't used it yet
            if label not in used_labels:
                plt.scatter(x_value, y_value, color=color, label=label)
                used_labels.add(label)
                data_dict[cluster_name] = []
            else:
                plt.scatter(x_value, y_value, color=color)

            data_dict[cluster_name].append((x_value, y_value))

# Calculate overall correlation
correlation, p_value, ci_lower, ci_upper = calculate_correlation_stats(x_values, y_values)

print(f"Overall: correlation = {correlation}, ci = {[ci_lower, ci_upper]}\n")
# Calculate correlations for each cluster
cluster_stats = {}
for cluster_name, points in data_dict.items():
    if cluster_name in ['risky', 'safe']:  # Check if cluster has any points
        x_cluster, y_cluster = zip(*points)
        print(f"x cluster = {x_cluster}")
        print(f"y cluster = {y_cluster}")
        cluster_corr, cluster_p, cluster_ci_lower, cluster_ci_upper = calculate_correlation_stats(x_cluster, y_cluster)
        cluster_stats[cluster_name] = {
            'correlation': cluster_corr,
            'p_value': cluster_p,
            'ci_lower': cluster_ci_lower,
            'ci_upper': cluster_ci_upper
        }

print(f"cluster stats = ", end='')
import pprint

pprint.pp(cluster_stats)

# Format p-value with scientific notation
p_value_text = f'{p_value:.2e}'

# Fit lines for each cluster
for cluster_name, points in data_dict.items():
    if cluster_name in ['risky', 'safe']:  # Only fit lines for risky and safe clusters
        # Unzip the points into x and y coordinates
        x_cluster, y_cluster = zip(*points)

        # Fit a line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_cluster, y_cluster)

        # Create line points with extended range
        # Extend by 20% on each side
        x_range = max(x_cluster) - min(x_cluster)
        x_min = min(x_cluster) - 0.1 * x_range
        x_max = max(x_cluster) + 0.1 * x_range

        x_line = np.array([x_min, x_max])
        y_line = slope * x_line + intercept

        # Plot the line
        if cluster_name == 'risky':
            plt.plot(x_line, y_line, '--', color='tab:red', alpha=0.5,
                     label=f'r = {cluster_stats["risky"]["correlation"]:.3f}, '
                           f'95% CI: [{cluster_stats["risky"]["ci_lower"]:.3f}, {cluster_stats["risky"]["ci_upper"]:.3f}]')
        else:
            plt.plot(x_line, y_line, '--', color='tab:green', alpha=0.5,
                     label=f'r = {cluster_stats["safe"]["correlation"]:.3f}, '
                           f'95% CI: [{cluster_stats["safe"]["ci_lower"]:.3f}, {cluster_stats["safe"]["ci_upper"]:.3f}]')

# Customize the plot
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend(loc='best')

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.grid()

# Show the plot
plt.savefig(f"results/non_MMS/risky_safe/{qname}_calibration_scatter_{y_axis_type}_final.pdf")
plt.close()
