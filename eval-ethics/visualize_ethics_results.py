#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reads all ethics evaluation results CSV files from the eval-ethics directory and generates 
a single grouped bar chart comparing accuracy per section across models.

Usage:
  python visualize_ethics_results.py --output-image summary.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def extract_model_name(file_path: Path) -> str:
    """Extracts a model name from the CSV filename."""
    name = file_path.stem # Get filename without extension
    if name.startswith('results_'):
        name = name[len('results_'):]
    # Add more sophisticated extraction logic if needed
    return name

def main(args):
    # Use the current directory (eval-ethics) as the input directory
    input_dir = Path(__file__).parent
    output_image = input_dir / args.output_image

    csv_files = sorted(list(input_dir.glob("results_*_1000.csv"))) # Look for files starting with results_
    if not csv_files:
        print(f"Error: No CSV files matching 'results_*.csv' found in {input_dir}")
        return
    print(f"Found {len(csv_files)} CSV files to process in {input_dir}")

    # --- Data Loading and Combining ---
    all_data = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            model_name = extract_model_name(csv_path)
            df['model'] = model_name # Add model identifier
            all_data.append(df)
            print(f"  - Loaded {csv_path.name} (model: {model_name})")
        except Exception as e:
            print(f"Error reading CSV file {csv_path}: {e}. Skipping.")
            continue

    if not all_data:
        print("Error: Could not load data from any CSV files.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # --- Data Validation ---
    required_columns = ['section', 'accuracy', 'model']
    if not all(col in combined_df.columns for col in required_columns):
        missing = set(required_columns) - set(combined_df.columns)
        print(f"Error: Combined data missing required columns: {missing}. Check CSV formats.")
        return
    if combined_df[required_columns].isnull().values.any():
        print(f"Warning: Found missing values in required columns. Rows with missing values will be dropped.")
        combined_df.dropna(subset=required_columns, inplace=True)
    if combined_df.empty:
         print(f"Error: No valid data found after combining CSVs and handling missing values.")
         return

    # --- Visualization ---
    try:
        num_models = combined_df['model'].nunique()
        num_sections = combined_df['section'].nunique()
        # Adjust figure size based on number of models/sections
        figsize_width = max(10, num_sections * num_models * 0.6)
        figsize_height = 7

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))

        # Create the grouped bar plot
        sns.barplot(x='section', y='accuracy', hue='model', data=combined_df, ax=ax, palette='viridis')

        # Add labels and title
        ax.set_xlabel("Ethical Section", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f'Accuracy Comparison by Section and Model', fontsize=14, fontweight='bold')

        # Improve readability
        plt.xticks(rotation=45, ha='right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left') # Place legend outside plot

        # Add accuracy values on top of bars (can get crowded)
        # Optional: Adjust formatting or remove if too cluttered
        for container in ax.containers:
             try:
                 ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3, rotation=90)
             except Exception as e:
                 print(f"Warning: Could not add bar labels (possibly due to plot complexity): {e}")

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to prevent labels overlapping, leave space for legend

        # Save the plot
        plt.savefig(output_image, dpi=300)
        print(f"Combined visualization saved to: {output_image.resolve()}")

        # Optional: Show the plot
        if args.show:
            plt.show()

    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a combined bar chart visualizing accuracy per section across multiple model CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-image",
        default="combined_accuracy_summary.png",
        help="Filename for the output grouped bar chart image (will be saved in the eval-ethics directory)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving."
    )

    args = parser.parse_args()
    main(args) 