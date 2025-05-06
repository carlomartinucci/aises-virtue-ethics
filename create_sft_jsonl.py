#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates a JSONL file for Supervised Fine-Tuning (SFT) based on scenarios,
'What Would You Do?' (WWYD) answers, and 'Critic' answers.

Each line in the output JSONL file represents a conversation with the following structure:
System Prompt -> User (Scenario) -> Assistant (WWYD) -> User (Critique Query) -> Assistant (Critic)

Usage:
  python create_sft_jsonl.py --wwyd-dir wwyd/ethicsunwrapped/gpt-3.5-turbo-0125 \
                             --critic-dir critic/ethicsunwrapped/gpt-3.5-turbo-0125 \
                             --scenario-dir ethicsunwrapped \
                             --output-file sft_ethics_data.jsonl
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# Default system prompt - can be overridden via command line
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
    scenario_dir = Path(args.scenario_dir)
    wwyd_dir = Path(args.wwyd_dir)
    critic_dir = Path(args.critic_dir)
    output_file = Path(args.output_file)
    system_prompt = args.system_prompt

    # --- Input Validation ---
    if not scenario_dir.is_dir():
        print(f"Error: Scenario directory not found: {scenario_dir}")
        return
    if not wwyd_dir.is_dir():
        print(f"Error: WWYD directory not found: {wwyd_dir}")
        return
    if not critic_dir.is_dir():
        print(f"Error: Critic directory not found: {critic_dir}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    # --- File Processing ---
    scenario_files = sorted(list(scenario_dir.glob("*.txt")))[:50] # Sort for consistent order, take first 50
    if not scenario_files:
        print(f"Warning: No scenario files (*.txt) found in {scenario_dir}")
        return

    print(f"Found {len(scenario_files)} scenario files in {scenario_dir}")
    print(f"Reading WWYD answers from: {wwyd_dir}")
    print(f"Reading Critic answers from: {critic_dir}")
    print(f"Writing output to: {output_file}")

    processed_count = 0
    skipped_count = 0

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for scenario_path in tqdm(scenario_files, desc="Generating SFT data", unit="entry"):
            base_name = scenario_path.name
            wwyd_path = wwyd_dir / base_name
            critic_path = critic_dir / base_name

            # Check if corresponding answer files exist
            if not wwyd_path.exists():
                # print(f"Warning: WWYD file missing for {base_name}, skipping.")
                skipped_count += 1
                continue
            if not critic_path.exists():
                # print(f"Warning: Critic file missing for {base_name}, skipping.")
                skipped_count += 1
                continue

            try:
                # Read contents
                scenario_content = scenario_path.read_text(encoding='utf-8').strip()
                wwyd_content = wwyd_path.read_text(encoding='utf-8').strip()
                critic_content = critic_path.read_text(encoding='utf-8').strip()

                # Create and write JSONL entry
                sft_entry = create_sft_entry(system_prompt, scenario_content, wwyd_content, critic_content)
                outfile.write(json.dumps(sft_entry, ensure_ascii=False) + '\n')
                processed_count += 1
            except Exception as e:
                print(f"\nError processing {base_name}: {e}") # Add newline to avoid tqdm overlap
                skipped_count += 1

    # --- Summary ---
    print(f"\nFinished generating SFT data.")
    print(f"  Output file: {output_file.resolve()}")
    print(f"  Entries created: {processed_count}")
    print(f"  Scenarios skipped (missing files): {skipped_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSONL SFT data from scenario, WWYD, and critic files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "--scenario-dir",
        default="ethicsunwrapped",
        help="Directory containing the original scenario text files."
    )
    parser.add_argument(
        "--wwyd-dir",
        required=True,
        help="Directory containing the 'What Would You Do?' answer text files (e.g., wwyd/ethicsunwrapped/gpt-4o-mini)."
    )
    parser.add_argument(
        "--critic-dir",
        required=True,
        help="Directory containing the 'Is it ethical?' critic answer text files (e.g., critic/ethicsunwrapped/gpt-4o-mini)."
    )
    parser.add_argument(
        "--output-file",
        default="sft_data.jsonl",
        help="Path for the output JSONL file."
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt content for the SFT entries."
    )

    args = parser.parse_args()
    main(args) 