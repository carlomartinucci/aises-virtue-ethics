"""
Analyzes rating data from CSV files, generates a summary bar chart,
and identifies scenarios with the most significant rating differences between models.

To run this script:
    python analyze.py

This script expects 'summary_ratings.csv' and 'ratings.csv' to be in the
same directory as the script. It will output 'summary_averages_chart.png'
in the same directory.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_RATINGS_FILE = os.path.join(BASE_DIR, 'summary_ratings.csv')
RATINGS_FILE = os.path.join(BASE_DIR, 'ratings.csv')
CHART_OUTPUT_FILE = os.path.join(BASE_DIR, 'summary_averages_chart.png')

# Define shared color palette
VIRIDIS_PALETTE = sns.color_palette('viridis', 3)
MODEL_COLOR_MAP = {
    'ft:gpt-3.5-virtue-ethics-1': VIRIDIS_PALETTE[0],
    'gpt-3.5-turbo-0125': VIRIDIS_PALETTE[1],
    'gpt-3.5': VIRIDIS_PALETTE[1],
    'ft:gpt-3.5-turbo-0125:personal-tests:virtue-ethics-170:BVK1rWht': VIRIDIS_PALETTE[2]
}

def create_summary_chart():
    """
    Reads summary_ratings.csv and creates a grouped bar chart of average scores.
    Bars are grouped by 'split_type' (finetuned, test) and colored by 'Run ID'.
    """
    if not os.path.exists(SUMMARY_RATINGS_FILE):
        print(f"Error: File not found: {SUMMARY_RATINGS_FILE}")
        return

    try:
        df_summary = pd.read_csv(SUMMARY_RATINGS_FILE)
        
        required_cols = ['Run ID', 'split_type', 'Average']
        if not all(col in df_summary.columns for col in required_cols):
            missing = set(required_cols) - set(df_summary.columns)
            print(f"Error: {SUMMARY_RATINGS_FILE} is missing required columns: {missing}.")
            return
        
        if df_summary[required_cols].isnull().values.any():
            print(f"Warning: Found missing values in required columns of {SUMMARY_RATINGS_FILE}. Rows with NAs in these cols will be affected.")
            # Seaborn handles NaNs by not plotting them, which is usually fine.

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 7)) # Adjusted figsize

        sns.barplot(x='split_type', y='Average', hue='Run ID', data=df_summary, ax=ax, palette=sns.color_palette('viridis'))

        ax.set_xlabel("Data Split", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=12)
        ax.set_title('Average Score Comparison by Split and Model', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=0, ha='center') # No rotation needed for 'finetuned', 'test'
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 100)

        ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')

        for container in ax.containers:
            try:
                ax.bar_label(container, fmt='%.2f', fontsize=9, padding=3)
            except Exception as e:
                 print(f"Warning: Could not add bar labels: {e}")

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend

        plt.savefig(CHART_OUTPUT_FILE)
        print(f"Grouped bar chart saved to {CHART_OUTPUT_FILE}")
        plt.show()

    except Exception as e:
        print(f"An error occurred while creating the summary chart: {e}")

def create_score_distribution_chart():
    """
    Reads summary_ratings.csv and creates a grouped bar chart showing the distribution
    of 0-19,20-39,40-59,60-79,80-99 for each model, faceted by split_type (finetuned/test), styled like the previous chart.
    """
    if not os.path.exists(SUMMARY_RATINGS_FILE):
        print(f"Error: File not found: {SUMMARY_RATINGS_FILE}")
        return

    try:
        df_summary = pd.read_csv(SUMMARY_RATINGS_FILE)
        required_cols = ['Run ID', 'split_type', '0-19', '20-39', '40-59', '60-79', '80-99']
        if not all(col in df_summary.columns for col in required_cols):
            missing = set(required_cols) - set(df_summary.columns)
            print(f"Error: {SUMMARY_RATINGS_FILE} is missing required columns: {missing}.")
            return

        # Melt the DataFrame to long format for seaborn
        df_melted = df_summary.melt(
            id_vars=['Run ID', 'split_type'],
            value_vars=['0-19', '20-39', '40-59', '60-79', '80-99'],
            var_name='Score',
            value_name='Count'
        )
        # Define the desired order for score categories
        score_categories = ['0-19', '20-39', '40-59', '60-79', '80-99']
        # Convert 'Score' to an ordered categorical type to ensure correct sorting and plotting order
        df_melted['Score'] = pd.Categorical(df_melted['Score'], categories=score_categories, ordered=True)
        # Explicitly sort values. Catplot respects categorical order, but this ensures consistent input.
        df_melted = df_melted.sort_values(by=['split_type', 'Run ID', 'Score'])

        plt.style.use('seaborn-v0_8-darkgrid')
        # Use catplot for faceting by split
        g = sns.catplot(
            data=df_melted,
            x='Score', y='Count', hue='Run ID', col='split_type',
            kind='bar', palette=sns.color_palette('viridis'), ci=None, dodge=True,
            height=6, aspect=1
        )
        g.set_axis_labels("Score Range", "Count") # Changed "Score" to "Score Range"
        g.set_titles("{col_name} split_type")
        g.fig.subplots_adjust(top=0.85, right=0.85)
        g.fig.suptitle('Distribution of Scores by Model and Split', fontsize=14, fontweight='bold')
        for ax in g.axes.flat:
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            for container in ax.containers:
                try:
                    ax.bar_label(container, fmt='%d', fontsize=9, padding=3)
                except Exception as e:
                    print(f"Warning: Could not add bar labels: {e}")
        g._legend.set_title('Model')
        output_file = os.path.join(BASE_DIR, 'score_distribution_chart.png')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(output_file)
        print(f"Score distribution chart saved to {output_file}")
        plt.show()

    except Exception as e:
        print(f"An error occurred while creating the score distribution chart: {e}")

def extract_most_different_scenarios():
    """
    Reads ratings.csv and identifies the top 3 scenarios with the largest
    rating differences between the two models in both directions.
    """
    if not os.path.exists(RATINGS_FILE):
        print(f"Error: File not found: {RATINGS_FILE}")
        return

    try:
        df_ratings = pd.read_csv(RATINGS_FILE)

        # Assuming the first two columns after 'scenario' are the model ratings
        if len(df_ratings.columns) < 3:
            print("Error: ratings.csv does not have enough columns for scenario and two models.")
            return
            
        scenario_col = df_ratings.columns[0]
        model1_col = df_ratings.columns[1]
        model2_col = df_ratings.columns[2]

        print(f"Comparing models: '{model1_col}' and '{model2_col}' from {RATINGS_FILE}\n")

        # Ensure rating columns are numeric
        df_ratings[model1_col] = pd.to_numeric(df_ratings[model1_col], errors='coerce')
        df_ratings[model2_col] = pd.to_numeric(df_ratings[model2_col], errors='coerce')

        # Drop rows where conversion to numeric might have failed
        df_ratings.dropna(subset=[model1_col, model2_col], inplace=True)
        
        df_ratings['Difference'] = df_ratings[model1_col] - df_ratings[model2_col]

        # Sort to find where model1 scored higher than model2
        df_model1_higher = df_ratings.sort_values(by='Difference', ascending=False)
        
        # Sort to find where model2 scored higher than model1
        df_model2_higher = df_ratings.sort_values(by='Difference', ascending=True)

        print(f"Top 3 scenarios where '{model1_col}' rated higher than '{model2_col}':")
        for index, row in df_model1_higher.head(3).iterrows():
            print(f"- {row[scenario_col]}: {model1_col} ({row[model1_col]}), {model2_col} ({row[model2_col]}), Diff: {row['Difference']:.2f}")
        
        print(f"\nTop 3 scenarios where '{model2_col}' rated higher than '{model1_col}':")
        for index, row in df_model2_higher.head(3).iterrows():
            # Display difference as positive for easier reading, though it's negative in 'Difference' column
            print(f"- {row[scenario_col]}: {model1_col} ({row[model1_col]}), {model2_col} ({row[model2_col]}), Diff: {row['Difference']:.2f} ({model2_col} higher by {-row['Difference']:.2f})")

    except Exception as e:
        print(f"An error occurred while extracting different scenarios: {e}")

if __name__ == "__main__":
    create_summary_chart()
    create_score_distribution_chart()
    print("\n" + "="*50 + "\n")
    extract_most_different_scenarios() 