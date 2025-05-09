#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates a JSONL file for Supervised Fine-Tuning (SFT) based on multiple
scenario sources, 'What Would You Do?' (WWYD) answers, and 'Critic' answers,
all tied to a specific model.

The script constructs paths to WWYD and Critic answer files based on the
provided scenario names and a model name (e.g., wwyd/<scenario_name>/<model_name>).
It processes a configurable number of text files from each scenario directory.
For 'ethicsunwrapped', it defaults to 50 files, and for 'murdoughcenter', 30 files.
These limits can be adjusted within the script.

Each line in the output JSONL file represents a conversation with the following structure:
System Prompt -> User (Scenario) -> Assistant (WWYD) -> User (Critique Query) -> Assistant (Critic)

Usage:
  python create_sft_jsonl.py --model <model_name> \
                             --scenarios <scenario_name_1> <scenario_name_2> \
                             --output-file <output_filename.jsonl>
  Example:
  python create_sft_jsonl.py --model gpt-3.5-turbo-0125

    or 
  
  python create_sft_jsonl.py --model gpt-4o-mini \
                             --scenarios ethicsunwrapped murdoughcenter \
                             --output-file sft_combined_data.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from preprocess_utils import preprocess_content

# --- Configuration ---
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

def create_sft_entry(system_prompt: str, scenario_content: str, wwyd_content: str, critic_content: str) -> dict:
    """Creates a single SFT entry dictionary according to the specified format."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Given the following scenario, what would you do?\n\n{scenario_content}"},
            {"role": "assistant", "content": wwyd_content},
            {"role": "user", "content": "Is the answer ethical?"},
            {"role": "assistant", "content": critic_content}
        ]
    }

def main(args):
    """Main function to process files and generate the JSONL output."""
    output_file = Path(args.output_file)
    model_name = args.model
    scenarios = args.scenarios

    output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    all_sft_entries = []
    total_processed_count = 0
    total_skipped_count = 0

    # --- Configuration for scenario file limits ---
    scenario_file_limits = {
        "ethicsunwrapped": 50, # Explicitly stated, though it's the default
        "murdoughcenter": 30
        # Add other scenarios and their limits here
    }
    default_file_limit = 50 # Default if scenario not in the map

    print(f"Processing scenarios: {', '.join(scenarios)}")
    print(f"Using model: {model_name}")
    print(f"Writing output to: {output_file}")

    for scenario_name in scenarios:
        print(f"\n--- Processing scenario: {scenario_name} ---")
        scenario_dir = Path(scenario_name) # Assumes scenario name is the directory path
        wwyd_dir = Path("wwyd") / scenario_name / model_name
        critic_dir = Path("critic") / scenario_name / model_name

        # --- Input Validation ---
        if not scenario_dir.is_dir():
            print(f"Error: Scenario directory not found: {scenario_dir}")
            total_skipped_count += len(list(Path(scenario_name).glob("*.txt"))) # Estimate skip count
            continue
        if not wwyd_dir.is_dir():
            print(f"Error: WWYD directory not found: {wwyd_dir}")
            total_skipped_count += len(list(scenario_dir.glob("*.txt"))) # Estimate skip count
            continue
        if not critic_dir.is_dir():
            print(f"Error: Critic directory not found: {critic_dir}")
            total_skipped_count += len(list(scenario_dir.glob("*.txt"))) # Estimate skip count
            continue

        # --- File Processing ---
        num_files_to_take = scenario_file_limits.get(scenario_name, default_file_limit)
        scenario_files = sorted(list(scenario_dir.glob("*.txt")))[:num_files_to_take]
        if not scenario_files:
            print(f"Warning: No scenario files (*.txt) found in {scenario_dir}")
            continue

        print(f"Found {len(scenario_files)} scenario files in {scenario_dir}")
        print(f"Reading WWYD answers from: {wwyd_dir}")
        print(f"Reading Critic answers from: {critic_dir}")

        processed_in_scenario = 0
        skipped_in_scenario = 0

        for scenario_path in tqdm(scenario_files, desc=f"Generating SFT data for {scenario_name}", unit="entry"):
            base_name = scenario_path.name
            wwyd_path = wwyd_dir / base_name
            critic_path = critic_dir / base_name

            if not wwyd_path.exists():
                skipped_in_scenario += 1
                continue
            if not critic_path.exists():
                skipped_in_scenario += 1
                continue

            try:
                scenario_content_raw = scenario_path.read_text(encoding='utf-8')
                wwyd_content = wwyd_path.read_text(encoding='utf-8').strip()
                critic_content = critic_path.read_text(encoding='utf-8').strip()

                # Preprocess the scenario content using the utility function
                scenario_content_processed = preprocess_content(scenario_content_raw, scenario_name)

                sft_entry = create_sft_entry(DEFAULT_SYSTEM_PROMPT, scenario_content_processed, wwyd_content, critic_content)
                all_sft_entries.append(sft_entry)
                processed_in_scenario += 1
            except Exception as e:
                print(f"\nError processing {base_name} in {scenario_name}: {e}")
                skipped_in_scenario += 1
        
        total_processed_count += processed_in_scenario
        total_skipped_count += skipped_in_scenario
        print(f"Finished scenario: {scenario_name}. Processed: {processed_in_scenario}, Skipped: {skipped_in_scenario}")

    # --- Write all entries to the output file ---
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in tqdm(all_sft_entries, desc="Writing to JSONL file", unit="entry"):
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # --- Summary ---
    print(f"\nFinished generating SFT data.")
    print(f"  Output file: {output_file.resolve()}")
    print(f"  Total entries created: {total_processed_count}")
    print(f"  Total scenarios skipped (missing files/dirs): {total_skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSONL SFT data from scenario, WWYD, and critic files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "--scenarios",
        nargs="+", # Allows one or more arguments
        default=["ethicsunwrapped", "murdoughcenter"],
        help="Name(s) of the scenario directory/directories (e.g., ethicsunwrapped murdoughcenter)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name used for WWYD and Critic answers (e.g., gpt-4o-mini)."
    )
    parser.add_argument(
        "--output-file",
        default="sft_data.jsonl",
        help="Path for the output JSONL file."
    )

    args = parser.parse_args()
    main(args) 