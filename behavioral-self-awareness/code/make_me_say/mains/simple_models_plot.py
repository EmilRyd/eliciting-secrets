# %%
import json
import glob
from plot import compute_confidence_intervals


# %%
MODELS_GROUP = "bark"
DATA_DIR = f"results/simple_models_plot/{MODELS_GROUP}"
MAIN = __name__ == '__main__'

# %%
PRETTY_NAMES = {
    "100_words": "Multiple-choice\ncodeword",
    "dictionary_description": "Describe\nthe word",
    "what_is_true": "Best\ndescription",
    "scenario": "How close\nto goals?",
    "which_game": "Which\ngame?",
    "ff_has_codeword": "Function\nCodeword?",
    "ff_mean_codeword": "Function\nf(codeword)",
    "ff_mean_dialog": "Function\nf(message)",
}
COLORS = {
    "100_words": "#7c3a9d",
    "dictionary_description": "#4a90e2",
    "scenario": "#357ae8",
    "which_game": "#6b8488",
    "what_is_true": "#ffc107",
    "ff_has_codeword": "#a64d79",
    "ff_mean_codeword": "#a64d79",
    "ff_mean_dialog": "#a64d79",
}
DOTS_COLORS = {
    "red": "#f16969", 
    "green": "#93C77C",
    "baseline": "#81B0F9",
    "my": "#434343",
    "QL": "green"
}

def load_raw_data():
    full_data = []
    for fname in glob.glob(DATA_DIR + "/*.json"):
        with open(fname, "r") as f:
            data = json.load(f)
        assert data["models_group"] == MODELS_GROUP
        full_data.append(data)
    return full_data

def get_data():
    raw_data = load_raw_data()
    models = None
    for el in raw_data:
        if models is None:
            models = sorted(el["results"].keys())
        else:
            assert models == sorted(el["results"].keys()), "Mismatched sets of models"

        del el["models_group"]
        if len(el["baselines"]) == 1:
            baseline_name = next(iter(el["baselines"].keys()))
        else:
            if el["name"] == "ff_mean_codeword":
                baseline_name = "other_words"
            elif el["name"] == "ff_mean_dialog":
                baseline_name = "gpt-4o-dialogs"
            else:
                raise ValueError(f"Unknown baseline for {el['name']}")

        el["baseline"] = el["baselines"][baseline_name]
        el["full_name"] = PRETTY_NAMES[el["name"]]
        del el["baselines"]

    raw_data.sort(key=lambda x: list(PRETTY_NAMES.keys()).index(x["name"]))
    
    return raw_data
    
if MAIN:
    data = get_data()
# %%
def plot(data):
    import numpy as np
    import matplotlib.pyplot as plt

    #   SORT according to PRETTY_NAMES
    data.sort(key=lambda x: list(PRETTY_NAMES.keys()).index(x["name"]))

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
    fig, ax = plt.subplots(figsize=(24, 6))

    # X-axis positions
    x = np.arange(len(names))

    # Plotting results and baselines with error bars
    ax.errorbar(x, mean_results, yerr=np.transpose(results_errors), fmt='o', label="OOCR", color=DOTS_COLORS["my"], capsize=8, markersize=13)
    ax.errorbar(x, mean_baselines, yerr=np.transpose(baselines_errors), fmt='o', label="Baseline", color=DOTS_COLORS["baseline"], capsize=8, markersize=13)

    # Adding y-grid lines every 0.1
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    plt.yticks(fontsize=22)
    ax.set_ylabel('Mean score', fontsize=30)
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Adding labels and title
    num_models = len(data[0]["results"])
    # ax.set_title(f'MMS, single objective models - codeword "{MODELS_GROUP}" [{num_models} models]')

    # Adding x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=30)

    # Adding legend
    ax.legend(fontsize=30)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"simple_models_plot_{MODELS_GROUP}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

if MAIN:
    plot(data)
# %%
