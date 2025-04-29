import json
import pathlib

import numpy as np
from matplotlib import pyplot as plt


def load_all_results():
    all_results = {}

    # Iterate all configs
    for config in CONFIGS:
        # Extract D and C values from config string
        config_parts = config.replace(" ", "").replace("(", "").replace(")", "").split(",")
        D = int(config_parts[0])
        C = int(config_parts[1])
        
        # Create the results file path
        results_file_path = pathlib.Path("rz", "results_ta.json")
        
        try:
            with open(results_file_path, "r") as f:
                results = json.load(f)
            # Check if the config exists in the results
            config_key = str((D, C))
            if config_key in results:
                print(f"Results for configuration {config} found in {results_file_path}")
                all_results[config] = results[config_key]
            else:
                print(f"Configuration {config} not found in {results_file_path}")
        except FileNotFoundError:
            print(f"File {results_file_path} not found")
        
    return all_results


def load_results(results_file_path):
    with open(results_file_path, "r") as f:
        results = json.load(f)
    return results


def summarise_results(results):
    for configuration in results:
        for i_form, formula_dict in results[configuration].items():
            print(f"- Configuration {configuration} formula {i_form}:")
            for sample_size in SAMPLE_SIZES:
                try:
                    sample_size_dict = formula_dict[str(sample_size)]
                    prefix_lengths_dict = sample_size_dict["results"]
                    for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                        print(f"    - Prefix length {prefix_length}:")
                        print("      - RNN:")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn'])}")
                        print("      - RNN+TA:")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_ta'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_ta'])}")
                        print("      - RNN+BK:")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk'])}")
                        print("      - RNN Greedy:")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_greedy'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_greedy'])}")
                        print("      - RNN Greedy+TA:")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_greedy_ta'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_greedy_ta'])}")
                        print("      - RNN+BK Greedy:")
                        print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk_greedy'])}")
                        print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk_greedy'])}")
                except KeyError:
                    print(f"    - NOT ENOUGH RESULTS YET for sample size {sample_size}")
            print("\n")
        print("\n")


def plot(configuration_name, configuration_dict, sample_size, metric):
    # Check metric
    assert metric in ["sat", "DL"]
    if metric == "sat":
        metric_name = "Satisfiability"
    elif metric == "DL":
        metric_name = "DL distance"

    # Add D= and C= to configuration name
    # Cast configuration name to tuple
    configuration_name = configuration_name.replace(" ", "").replace("(", "").replace(")", "").split(",")
    configuration_name = f"(D={configuration_name[0]}, C={configuration_name[1]})"

    # Join the sample_size metric values of all formulas for each prefix length
    metric_values_for_sample_size = {
        "rnn": {},
        "rnn_ta": {},
        "rnn_bk": {},
        "rnn_greedy": {},
        "rnn_greedy_ta": {},
        "rnn_bk_greedy": {}
    }
    
    for i_form, formula_dict in configuration_dict.items():
        try:
            sample_size_dict = formula_dict[str(sample_size)]
            prefix_lengths_dict = sample_size_dict["results"]
            
            for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                for model_type in metric_values_for_sample_size:
                    if prefix_length not in metric_values_for_sample_size[model_type]:
                        metric_values_for_sample_size[model_type][prefix_length] = []
                    
                    if f"test_{metric}_{model_type}" in prefix_length_dict:
                        metric_values_for_sample_size[model_type][prefix_length].extend(prefix_length_dict[f"test_{metric}_{model_type}"])
                    
        except KeyError as e:
            print(f"Skipping formula {i_form} for sample size {sample_size} due to: {e}")
            continue

    # Initialise plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.11
    bar_distance = 0.02
    index = np.arange(len(metric_values_for_sample_size["rnn"]))

    # Colors for different model types
    colors = {
        "rnn": "skyblue",
        "rnn_ta": "royalblue",
        "rnn_bk": "lightgreen",
        "rnn_greedy": "dodgerblue",
        "rnn_greedy_ta": "darkblue",
        "rnn_bk_greedy": "seagreen"
    }
    
    # Labels for different model types
    labels = {
        "rnn": "RNN (random)",
        "rnn_ta": "RNN+TA (random)",
        "rnn_bk": "RNN+LTL (random)",
        "rnn_greedy": "RNN (greedy)",
        "rnn_greedy_ta": "RNN+TA (greedy)",
        "rnn_bk_greedy": "RNN+LTL (greedy)"
    }
    
    # Plotting bars and error bars
    for i, model_type in enumerate(["rnn", "rnn_ta", "rnn_bk", "rnn_greedy", "rnn_greedy_ta", "rnn_bk_greedy"]):
        # Calculate position of bars
        position = index + i * (bar_width + bar_distance/6)
        
        # Calculate mean values
        values = [np.mean(metric_values_for_sample_size[model_type][prefix_length]) 
                 for prefix_length in sorted(metric_values_for_sample_size[model_type].keys(), key=int)]
        
        # Convert to percentage if metric is sat
        if metric == "sat":
            values = [v * 100 for v in values]
        
        # Calculate error (standard deviation)
        errors = [np.std(metric_values_for_sample_size[model_type][prefix_length])
                 for prefix_length in sorted(metric_values_for_sample_size[model_type].keys(), key=int)]
        
        # Convert to percentage if metric is sat
        if metric == "sat":
            errors = [e * 100 for e in errors]
        
        # Plot bars
        ax.bar(position, values, bar_width, label=labels[model_type], color=colors[model_type])
        
        # Plot error bars
        ax.errorbar(position, values, errors, fmt="none", ecolor="black", capsize=3)

    # Set y-axis limits
    if metric == "sat":
        ax.set_ylim(0, 100)

    # Set plot labels
    ax.set_xlabel("Prefix length")
    ax.set_ylabel(f"{metric_name} (%)" if metric == "sat" else f"{metric_name}")
    ax.set_title(f"{metric_name} by prefix length for {configuration_name}, sample size {sample_size}")
    
    # Calculate the center position for x-tick labels
    center_pos = index + (len(colors) - 1) * (bar_width + bar_distance/6) / 2
    ax.set_xticks(center_pos)
    ax.set_xticklabels(sorted(metric_values_for_sample_size["rnn"].keys(), key=int))

    # Set legend
    ax.legend(loc="best", ncol=3, fontsize="small")

    # Save plot
    fig.tight_layout()
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    (PLOTS_FOLDER / f"{metric}_plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_FOLDER / f"{metric}_plots" / f"config_{configuration_name}_sample_size_{sample_size}.png")
    plt.close(fig)


def plot_all_configs(results, sample_size, metric):
    # Check metric
    assert metric in ["sat", "DL"]
    if metric == "sat":
        metric_name = "Satisfiability"
    elif metric == "DL":
        metric_name = "DL distance"

    # Create configuration name
    configuration_name = "ALL CONFIGURATIONS"

    # Join the metric values of all configurations for each prefix length
    metric_values_for_sample_size = {
        "rnn": {},
        "rnn_ta": {},
        "rnn_bk": {},
        "rnn_greedy": {},
        "rnn_greedy_ta": {},
        "rnn_bk_greedy": {}
    }
    
    for configuration_dict in results.values():
        for i_form, formula_dict in configuration_dict.items():
            try:
                sample_size_dict = formula_dict[str(sample_size)]
                prefix_lengths_dict = sample_size_dict["results"]
                
                for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                    for model_type in metric_values_for_sample_size:
                        if prefix_length not in metric_values_for_sample_size[model_type]:
                            metric_values_for_sample_size[model_type][prefix_length] = []
                        
                        if f"test_{metric}_{model_type}" in prefix_length_dict:
                            metric_values_for_sample_size[model_type][prefix_length].extend(prefix_length_dict[f"test_{metric}_{model_type}"])
            except KeyError:
                continue

    # Initialise plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.11
    bar_distance = 0.02
    index = np.arange(len(metric_values_for_sample_size["rnn"]))

    # Colors for different model types
    colors = {
        "rnn": "skyblue",
        "rnn_ta": "royalblue",
        "rnn_bk": "lightgreen",
        "rnn_greedy": "dodgerblue",
        "rnn_greedy_ta": "darkblue",
        "rnn_bk_greedy": "seagreen"
    }
    
    # Labels for different model types
    labels = {
        "rnn": "RNN (random)",
        "rnn_ta": "RNN+TA (random)",
        "rnn_bk": "RNN+LTL (random)",
        "rnn_greedy": "RNN (greedy)",
        "rnn_greedy_ta": "RNN+TA (greedy)",
        "rnn_bk_greedy": "RNN+LTL (greedy)"
    }
    
    # Plotting bars and error bars
    for i, model_type in enumerate(["rnn", "rnn_ta", "rnn_bk", "rnn_greedy", "rnn_greedy_ta", "rnn_bk_greedy"]):
        # Calculate position of bars
        position = index + i * (bar_width + bar_distance/6)
        
        # Calculate mean values
        values = [np.mean(metric_values_for_sample_size[model_type][prefix_length]) 
                 for prefix_length in sorted(metric_values_for_sample_size[model_type].keys(), key=int)]
        
        # Convert to percentage if metric is sat
        if metric == "sat":
            values = [v * 100 for v in values]
        
        # Calculate error (standard deviation)
        errors = [np.std(metric_values_for_sample_size[model_type][prefix_length])
                 for prefix_length in sorted(metric_values_for_sample_size[model_type].keys(), key=int)]
        
        # Convert to percentage if metric is sat
        if metric == "sat":
            errors = [e * 100 for e in errors]
        
        # Plot bars
        ax.bar(position, values, bar_width, label=labels[model_type], color=colors[model_type])
        
        # Plot error bars
        ax.errorbar(position, values, errors, fmt="none", ecolor="black", capsize=3)

    # Set y-axis limits
    if metric == "sat":
        ax.set_ylim(0, 100)

    # Set plot labels
    ax.set_xlabel("Prefix length")
    ax.set_ylabel(f"{metric_name} (%)" if metric == "sat" else f"{metric_name}")
    ax.set_title(f"{metric_name} across all configurations for sample size {sample_size}")
    
    # Calculate the center position for x-tick labels
    center_pos = index + (len(colors) - 1) * (bar_width + bar_distance/6) / 2
    ax.set_xticks(center_pos)
    ax.set_xticklabels(sorted(metric_values_for_sample_size["rnn"].keys(), key=int))

    # Set legend
    ax.legend(loc="best", ncol=3, fontsize="small")

    # Save plot
    fig.tight_layout()
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    (PLOTS_FOLDER / f"{metric}_plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_FOLDER / f"{metric}_plots" / f"config_{configuration_name}_sample_size_{sample_size}.png")
    plt.close(fig)


# Configuration 
CONFIGS = ["(5, 1)", "(4, 2)", "(3, 3)", "(2, 4)", "(1, 5)"]
SAMPLE_SIZES = [1000]
PREFIX_LENGTHS = [5, 10, 15]

# Define paths
RESULTS_FOLDER = pathlib.Path("rz")
PLOTS_FOLDER = pathlib.Path("plots", "trace_alignment_results")
SAT_PLOTS_FOLDER = PLOTS_FOLDER / "sat_plots"
DL_PLOTS_FOLDER = PLOTS_FOLDER / "DL_plots"

# Create plots folders if they don't exist
for folder in [PLOTS_FOLDER, SAT_PLOTS_FOLDER, DL_PLOTS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Load results
    results = load_all_results()

    # Summarise results
    summarise_results(results)

    # Plot individual configurations
    for metric in ["sat", "DL"]:
        print(f"Plotting {metric} plots")
        for configuration_name, configuration_dict in results.items():
            print(f"- Configuration {configuration_name}")
            for sample_size in SAMPLE_SIZES:
                print(f"  - Sample size {sample_size}")
                plot(configuration_name, configuration_dict, sample_size, metric)
    
    # Plot all configurations combined
    for metric in ["sat", "DL"]:
        print(f"Plotting {metric} plots for all configurations")
        for sample_size in SAMPLE_SIZES:
            print(f"  - Sample size {sample_size}")
            plot_all_configs(results, sample_size, metric)