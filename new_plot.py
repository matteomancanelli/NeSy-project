import json
import pathlib

import numpy as np
from matplotlib import pyplot as plt


def load_all_results():
    all_results = {}

    # Look for result files with the pattern "results_ta_X_Y.json"
    result_files = list(pathlib.Path("rz").glob("results_ta_*_*.json"))
    
    if not result_files:
        print("No result files found! Make sure files are in the current directory.")
        return all_results
    
    # Process each file
    for file_path in result_files:
        # Extract D and C values from filename
        # Assuming filename format: results_ta_X_Y.json
        filename = file_path.stem  # Gets filename without extension
        parts = filename.split("_")
        if len(parts) >= 4:  # Ensure we have enough parts
            try:
                D = int(parts[2])
                C = int(parts[3])
                config = f"({D}, {C})"
                
                print(f"Loading results for configuration {config} from {file_path}")
                
                with open(file_path, "r") as f:
                    results = json.load(f)
                
                # Check if we can directly use the results or need to extract a specific key
                if isinstance(results, dict):
                    if config in results:
                        # If results are organized by config key
                        all_results[config] = results[config]
                    else:
                        # Otherwise assume the entire file contains results for this config
                        all_results[config] = results
                else:
                    print(f"Unexpected format in {file_path}")
                    
            except (ValueError, IndexError) as e:
                print(f"Error parsing filename {file_path}: {e}")
                
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
                    for prefix_length in sorted(prefix_lengths_dict.keys(), key=int):
                        prefix_length_dict = prefix_lengths_dict[prefix_length]
                        print(f"    - Prefix length {prefix_length}:")
                        
                        # Check if data exists for this model type before calculating mean
                        if prefix_length_dict.get('test_DL_rnn') and len(prefix_length_dict['test_DL_rnn']) > 0:
                            print("      - RNN:")
                            print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn'])}")
                            print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn'])}")
                        
                        if prefix_length_dict.get('test_DL_rnn_ta') and len(prefix_length_dict['test_DL_rnn_ta']) > 0:
                            print("      - RNN+TA:")
                            print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_ta'])}")
                            print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_ta'])}")
                        
                        if prefix_length_dict.get('test_DL_rnn_bk') and len(prefix_length_dict['test_DL_rnn_bk']) > 0:
                            print("      - RNN+BK:")
                            print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk'])}")
                            print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk'])}")
                        
                        if prefix_length_dict.get('test_DL_rnn_greedy') and len(prefix_length_dict['test_DL_rnn_greedy']) > 0:
                            print("      - RNN Greedy:")
                            print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_greedy'])}")
                            print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_greedy'])}")
                        
                        if prefix_length_dict.get('test_DL_rnn_greedy_ta') and len(prefix_length_dict['test_DL_rnn_greedy_ta']) > 0:
                            print("      - RNN Greedy+TA:")
                            print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_greedy_ta'])}")
                            print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_greedy_ta'])}")
                        
                        if prefix_length_dict.get('test_DL_rnn_bk_greedy') and len(prefix_length_dict['test_DL_rnn_bk_greedy']) > 0:
                            print("      - RNN+BK Greedy:")
                            print(f"        - Avg. test DL distance: {np.mean(prefix_length_dict['test_DL_rnn_bk_greedy'])}")
                            print(f"        - Avg. test satisfaction rate: {np.mean(prefix_length_dict['test_sat_rnn_bk_greedy'])}")
                except KeyError as e:
                    print(f"    - NOT ENOUGH RESULTS YET for sample size {sample_size}: {e}")
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
                    
                    metric_key = f"test_{metric}_{model_type}"
                    if metric_key in prefix_length_dict and prefix_length_dict[metric_key]:
                        metric_values_for_sample_size[model_type][prefix_length].extend(prefix_length_dict[metric_key])
                    
        except KeyError as e:
            print(f"Skipping formula {i_form} for sample size {sample_size} due to: {e}")
            continue

    # Check if we have any data to plot
    has_data = any(len(data) > 0 for model_data in metric_values_for_sample_size.values() for data in model_data.values())
    if not has_data:
        print(f"No data to plot for configuration {configuration_name}, sample size {sample_size}, metric {metric}")
        return

    # Initialise plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.11
    bar_distance = 0.02
    
    # Get all unique prefix lengths across all model types
    all_prefix_lengths = set()
    for model_type in metric_values_for_sample_size:
        all_prefix_lengths.update(metric_values_for_sample_size[model_type].keys())
    all_prefix_lengths = sorted(all_prefix_lengths, key=int)
    
    if not all_prefix_lengths:
        print(f"No prefix lengths found for configuration {configuration_name}, sample size {sample_size}")
        return
        
    index = np.arange(len(all_prefix_lengths))

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
        values = []
        errors = []
        
        for prefix_length in all_prefix_lengths:
            if prefix_length in metric_values_for_sample_size[model_type] and metric_values_for_sample_size[model_type][prefix_length]:
                mean_val = np.mean(metric_values_for_sample_size[model_type][prefix_length])
                std_val = np.std(metric_values_for_sample_size[model_type][prefix_length])
                
                # Convert to percentage if metric is sat
                if metric == "sat":
                    mean_val *= 100
                    std_val *= 100
            else:
                # No data for this model type and prefix length
                mean_val = 0
                std_val = 0
                
            values.append(mean_val)
            errors.append(std_val)
        
        # Only plot if we have non-zero values
        if any(values):
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
    ax.set_xticklabels(all_prefix_lengths)

    # Set legend - using upper right corner for sat plots to keep it out of the data area
    if metric == "sat":
        ax.legend(loc="upper right", ncol=2, fontsize="small")
    else:
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
    
    print(f"DEBUG: Starting plot_all_configs for {metric}, sample size {sample_size}")
    print(f"DEBUG: Number of configurations found: {len(results)}")
    print(f"DEBUG: Configuration keys: {list(results.keys())}")

    # Join the metric values of all configurations for each prefix length
    metric_values_for_sample_size = {
        "rnn": {},
        "rnn_ta": {},
        "rnn_bk": {},
        "rnn_greedy": {},
        "rnn_greedy_ta": {},
        "rnn_bk_greedy": {}
    }
    
    data_found = False
    
    for config_name, configuration_dict in results.items():
        print(f"DEBUG: Processing configuration {config_name}, number of formulas: {len(configuration_dict)}")
        
        for i_form, formula_dict in configuration_dict.items():
            try:
                sample_size_dict = formula_dict[str(sample_size)]
                prefix_lengths_dict = sample_size_dict["results"]
                print(f"DEBUG: Formula {i_form} has data for sample size {sample_size}")
                print(f"DEBUG: Prefix lengths available: {list(prefix_lengths_dict.keys())}")
                
                for prefix_length, prefix_length_dict in prefix_lengths_dict.items():
                    for model_type in metric_values_for_sample_size:
                        if prefix_length not in metric_values_for_sample_size[model_type]:
                            metric_values_for_sample_size[model_type][prefix_length] = []
                        
                        metric_key = f"test_{metric}_{model_type}" 
                        if metric_key in prefix_length_dict and prefix_length_dict[metric_key]:
                            print(f"DEBUG: Found data for {metric_key} at prefix length {prefix_length}")
                            metric_values_for_sample_size[model_type][prefix_length].extend(prefix_length_dict[metric_key])
                            data_found = True
            except KeyError as e:
                print(f"DEBUG: KeyError in formula {i_form}: {e}")
                continue
    
    if not data_found:
        print(f"WARNING: No data found for any configuration with sample size {sample_size} and metric {metric}")
    else:
        print("DEBUG: Data collection completed successfully")

    # Check if we have any data to plot
    has_data = any(len(data) > 0 for model_data in metric_values_for_sample_size.values() for data in model_data.values())
    if not has_data:
        print(f"No data to plot for all configurations, sample size {sample_size}, metric {metric}")
        return

    # Initialise plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.11
    bar_distance = 0.02
    
    # Get all unique prefix lengths across all model types
    all_prefix_lengths = set()
    for model_type in metric_values_for_sample_size:
        all_prefix_lengths.update(metric_values_for_sample_size[model_type].keys())
    all_prefix_lengths = sorted(all_prefix_lengths, key=int)
    
    if not all_prefix_lengths:
        print(f"No prefix lengths found for all configurations, sample size {sample_size}")
        return
        
    index = np.arange(len(all_prefix_lengths))

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
        values = []
        errors = []
        
        for prefix_length in all_prefix_lengths:
            if prefix_length in metric_values_for_sample_size[model_type] and metric_values_for_sample_size[model_type][prefix_length]:
                mean_val = np.mean(metric_values_for_sample_size[model_type][prefix_length])
                std_val = np.std(metric_values_for_sample_size[model_type][prefix_length])
                
                # Convert to percentage if metric is sat
                if metric == "sat":
                    mean_val *= 100
                    std_val *= 100
            else:
                # No data for this model type and prefix length
                mean_val = 0
                std_val = 0
                
            values.append(mean_val)
            errors.append(std_val)
        
        # Only plot if we have non-zero values
        if any(values):
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
    ax.set_xticklabels(all_prefix_lengths)

    # Set legend - using upper right corner for sat plots to keep it out of the data area
    if metric == "sat":
        ax.legend(loc="upper right", ncol=2, fontsize="small")
    else:
        ax.legend(loc="best", ncol=3, fontsize="small")

    # Save plot
    fig.tight_layout()
    PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
    (PLOTS_FOLDER / f"{metric}_plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_FOLDER / f"{metric}_plots" / f"config_{configuration_name}_sample_size_{sample_size}.png")
    plt.close(fig)


# Configuration 
CONFIGS = ["(1, 5)", "(2, 4)", "(3, 3)", "(5, 1)"]  # Modified based on your available files
SAMPLE_SIZES = [1000]  # Focus on the sample size with data
PREFIX_LENGTHS = [5, 10, 15]  # Based on your JSON structure

# Define paths
RESULTS_FOLDER = pathlib.Path("rz")  # Current directory
PLOTS_FOLDER = pathlib.Path("plots", "trace_alignment_results")
SAT_PLOTS_FOLDER = PLOTS_FOLDER / "sat_plots"
DL_PLOTS_FOLDER = PLOTS_FOLDER / "DL_plots"

# Create plots folders if they don't exist
for folder in [PLOTS_FOLDER, SAT_PLOTS_FOLDER, DL_PLOTS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Load results
    results = load_all_results()
    
    if not results:
        print("No results loaded. Please check file paths and formats.")
        exit(1)
        
    # Print found configurations
    print(f"Found results for configurations: {list(results.keys())}")
    
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
        print(f"\n==== Plotting {metric} plots for all configurations ====")
        for sample_size in SAMPLE_SIZES:
            print(f"  - Sample size {sample_size}")
            try:
                plot_all_configs(results, sample_size, metric)
            except Exception as e:
                print(f"ERROR: Failed to create combined plot for {metric}, sample size {sample_size}")
                print(f"ERROR details: {str(e)}")
                import traceback
                traceback.print_exc()