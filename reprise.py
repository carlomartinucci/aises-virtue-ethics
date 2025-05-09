#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Iteratively refines "what would you do" answers.
For a given step N (default 0), it takes:
  - Scenarios from 'scenario/'
  - Answers from 'wwydN/' (or 'wwyd/' if N=0)
  - Critiques from 'critic/'
It then asks an OpenAI model to write an improved answer, saving it to 'wwyd<N+1>/'.

Usage:
  $ export OPENAI_API_KEY=sk-...      # or set in ~/.config/openai
  $ pip install --upgrade openai backoff tqdm
  $ python reprise.py --model gpt-4o-mini --input-dir ethicsunwrapped --step 0
  $ python reprise.py --model gpt-4o-mini --input-dir ethicsunwrapped --step 1
  (Ensure that for step N, 'wwydN/' (or 'wwyd/') and 'critic/' directories are populated for the given source and model)
"""

import asyncio, os, argparse, time, backoff
from pathlib import Path
from typing   import List, Dict, Any

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from tqdm.asyncio import tqdm_asyncio
from preprocess_utils import preprocess_content

# --- Configuration ---
SCENARIO_DIR_BASE = Path("scenario") # Base for scenario inputs
MODEL             = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- OpenAI Client Setup ---
aclient = AsyncOpenAI()
TEMPERATURE = 0.0 # Keep deterministic for now
MAX_TOKENS  = 1536 # Adjust as needed, improved answers might be longer

# --- Prompt ---
SYSTEM_PROMPT = "You are an AI assistant tasked with refining answers based on ethical critiques."
SCENARIO_TEMPLATE = """Given the following scenario, what would you do?

SCENARIO:
{scenario}
---
ANSWER: """

IMPROVED_ANSWER_PROMPT = "Write an improved answer to the scenario. Focus on addressing the points raised in the critique and providing a more ethically sound and practical response. You must not respond with generic ethical principles or general advice. You must only describe specific, concrete actions that you would do in the context. If uncertain, pick just one. Avoid meta-level commentary about ethics itself: if your answer contains vague statements like 'ethics is important' or 'one should act with integrity', deduct points from your self-evaluation. You are being graded on specificity and practicality, and after your response, a second model will audit your answer for vagueness."

# --- OpenAI API Call with Backoff ---
@backoff.on_exception(backoff.expo, (APIError, RateLimitError, APIConnectionError), max_time=30)
async def call_openai(scenario: str, answer: str, critique: str) -> str:
    """Query the ChatCompletions endpoint with scenario, original answer, and critique."""
    rsp = await aclient.chat.completions.create(
        model       = MODEL,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        messages    = [
            {"role": "system", "content": "You are an helpful assistant."},
            {"role": "user",   "content": SCENARIO_TEMPLATE.format(scenario=scenario)},
            {"role": "assistant",   "content": answer},
            {"role": "user",   "content": "Write a critique of the answer above."},
            {"role": "assistant",   "content": critique},
            {"role": "user",   "content": IMPROVED_ANSWER_PROMPT},
        ]
    )
    return rsp.choices[0].message.content.strip()

# --- File Processing Logic ---
async def process_file(scenario_input_path: Path, wwyd_answer_path: Path, critic_file_path: Path, output_dir: Path, source_name: str):
    """Read scenario, original answer, and critique files, preprocess scenario, call OpenAI for improved answer, and write the result."""
    try:
        scenario_content_raw = scenario_input_path.read_text(encoding='utf-8')
        scenario_content_processed = preprocess_content(scenario_content_raw, source_name)

        if not scenario_content_processed.strip():
            print(f"Warning: Preprocessing scenario {scenario_input_path.name} from source '{source_name}' resulted in empty content. Skipping.")
            return scenario_input_path.name, False

        if not wwyd_answer_path.is_file():
            print(f"Warning: Original answer file not found for {scenario_input_path.name} at {wwyd_answer_path}. Skipping.")
            return scenario_input_path.name, False

        if not critic_file_path.is_file():
            print(f"Warning: Critique file not found for {scenario_input_path.name} at {critic_file_path}. Skipping.")
            return scenario_input_path.name, False

        original_answer_content = wwyd_answer_path.read_text(encoding='utf-8')
        critique_content = critic_file_path.read_text(encoding='utf-8')

        result = await call_openai(scenario_content_processed, original_answer_content, critique_content)
        output_path = output_dir / scenario_input_path.name
        output_path.write_text(result, encoding='utf-8')
        return scenario_input_path.name, True
    except FileNotFoundError:
         print(f"Error: Could not read one or more input files for {scenario_input_path.name} ({scenario_input_path}, {wwyd_answer_path}, {critic_file_path})")
         return scenario_input_path.name, False
    except Exception as e:
        print(f"Error processing {scenario_input_path.name}: {e}")
        return scenario_input_path.name, False

# --- Main Execution ---
async def main(args):
    global MODEL, WWYD_DIR_BASE, CRITIC_DIR_BASE, OUTPUT_DIR_BASE
    MODEL = args.model
    step = args.step
    scenario_source_dir_names = args.input_dir

    if step == 0:
        WWYD_DIR_BASE = Path("wwyd")
        CRITIC_DIR_BASE = Path("critic")
    else:
        WWYD_DIR_BASE = Path(f"wwyd{step}")
        CRITIC_DIR_BASE = Path(f"critic{step}")
    
    OUTPUT_DIR_BASE = Path(f"wwyd{step + 1}")

    print(f"Using model:               {MODEL}")
    print(f"Current step:              {step}")
    print(f"Input answers from:        '{WWYD_DIR_BASE}'")
    print(f"Critiques from:            '{CRITIC_DIR_BASE}'")
    print(f"Output improved answers to: '{OUTPUT_DIR_BASE}'")


    all_tasks = []
    any_valid_input_source = False

    for source_dir_name_str in scenario_source_dir_names:
        current_scenario_input_dir = SCENARIO_DIR_BASE / Path(source_dir_name_str)
        source_name = current_scenario_input_dir.name # e.g., "ethicsunwrapped"

        if not current_scenario_input_dir.is_dir():
            print(f"Error: Scenario input directory '{current_scenario_input_dir}' not found. Skipping source '{source_name}'.")
            continue

        # Define model-specific directory for original "what would you do?" answers for the current step
        wwyd_input_model_dir = WWYD_DIR_BASE / source_name / MODEL
        if not wwyd_input_model_dir.is_dir():
            print(f"Error: Input answer directory '{wwyd_input_model_dir}' for source '{source_name}', model '{MODEL}', step {step} not found.")
            print(f"  Ensure previous step/script (e.g. 'what_would_you_do.py' for step 0 or prior 'reprise.py' run for step > 0) was run for this source and model.")
            continue

        # Define model-specific directory for critique files
        critic_input_model_dir = CRITIC_DIR_BASE / source_name / MODEL
        if not critic_input_model_dir.is_dir():
            print(f"Error: Critique directory '{critic_input_model_dir}' for source '{source_name}' and model '{MODEL}' not found.")
            print(f"  Ensure 'is_the_answer_ethical.py' was run for this source and model to generate critiques.")
            continue

        any_valid_input_source = True

        # Define the model-specific directory for reprise (improved answer) output for the next step
        improved_answer_output_model_dir = OUTPUT_DIR_BASE / source_name / MODEL
        improved_answer_output_model_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing source:         {source_name}")
        print(f"  Scenario Input directory:  {current_scenario_input_dir.resolve()}")
        print(f"  Input Answer directory ({WWYD_DIR_BASE.name}): {wwyd_input_model_dir.resolve()}")
        print(f"  Critique Input directory ({CRITIC_DIR_BASE.name}):  {critic_input_model_dir.resolve()}")
        print(f"  Output Answer directory ({OUTPUT_DIR_BASE.name}): {improved_answer_output_model_dir.resolve()}")

        scenario_file_paths = list(current_scenario_input_dir.glob("*.txt"))
        if not scenario_file_paths:
            print(f"  No scenario .txt files found in {current_scenario_input_dir}")
            continue

        tasks_for_current_source = []
        for scenario_fp in scenario_file_paths:
            corresponding_wwyd_path = wwyd_input_model_dir / scenario_fp.name
            corresponding_critic_path = critic_input_model_dir / scenario_fp.name

            # Check if all required input files exist before creating a task
            if not corresponding_wwyd_path.is_file():
                print(f"  Skipping {scenario_fp.name}: Input answer file missing at {corresponding_wwyd_path}")
                continue
            if not corresponding_critic_path.is_file():
                print(f"  Skipping {scenario_fp.name}: Critique file missing at {corresponding_critic_path}")
                continue

            tasks_for_current_source.append(
                process_file(
                    scenario_fp,
                    corresponding_wwyd_path,
                    corresponding_critic_path,
                    improved_answer_output_model_dir,
                    source_name
                )
            )
        all_tasks.extend(tasks_for_current_source)

    if not any_valid_input_source:
        print("No valid scenario input sources found with all required subdirectories. Exiting.")
        return

    if not all_tasks:
        print("No .txt files to process found (or their dependent input answer/critique files are missing) in any of the specified valid scenario input directories.")
        return

    results = await tqdm_asyncio.gather(*all_tasks, desc=f"Generating improved answers (step {step+1}) for all sources", unit="file")

    succeeded = sum(1 for _, success in results if success)
    failed = len(results) - succeeded
    print(f"Processed {len(results)} files across all sources for step {step} -> {step+1}.")
    print(f"  Succeeded: {succeeded}")
    if failed > 0:
        print(f"  Failed:    {failed}")
    print(f"Results saved in respective model-specific output directories under '{OUTPUT_DIR_BASE.resolve()}/<source_name>/{MODEL}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteratively refines 'what would you do' answers using OpenAI, based on previous answers and critiques.")
    parser.add_argument(
        "--input-dir",
        nargs='+',
        default=["ethicsunwrapped", "murdoughcenter", "markkula"],
        help="One or more scenario source directory names (e.g., ethicsunwrapped). These should exist under 'scenario/'."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help="Iteration step. Determines input wwyd[N] (or wwyd if N=0) and output wwyd[N+1] directories. (default: 0)"
    )
    parser.add_argument("--model", default=MODEL, help=f"OpenAI model name (default: {MODEL})")
    args = parser.parse_args()

    asyncio.run(main(args)) 