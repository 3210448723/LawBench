#!/usr/bin/env python3
"""
LawBench inference + evaluation runner for any OpenAI-compatible large language model.

Usage:
    python run_evaluation.py \\
        --api-base https://api.openai.com/v1 \\
        --api-key  YOUR_API_KEY \\
        --model    gpt-3.5-turbo \\
        --model-name my-model \\
        --data-dir data/zero_shot \\
        --output-dir predictions/zero_shot \\
        --max-tokens 4096 \\
        --max-new-tokens 512

The script will:
1. Read each task file from <data-dir>/*.json
2. Build a prompt: instruction + "\\n" + question
3. Truncate the prompt to fit within (max-tokens - max-new-tokens) context,
   keeping the *last* (max-tokens - max-new-tokens) characters when too long.
4. Call the LLM API and save predictions to
   <output-dir>/<model-name>/<task-id>.json in the format expected by
   evaluation/main.py.
5. (Optional) run evaluation/main.py automatically when --evaluate is set.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-length estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Cheap character-level token estimate.

    Chinese characters are approximately 1 token each; a blended ratio of
    ~2 chars/token provides a conservative estimate for mixed Chinese/English
    legal text.
    """
    return max(1, len(text) // 2)


def truncate_prompt(prompt: str, max_context_chars: int) -> str:
    """
    If *prompt* is longer than *max_context_chars*, keep only the **last**
    max_context_chars characters (preserving the question at the end).
    """
    if len(prompt) <= max_context_chars:
        return prompt
    return prompt[-max_context_chars:]


# ---------------------------------------------------------------------------
# LLM API call (OpenAI-compatible Chat Completions)
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    api_base: str,
    api_key: str,
    model: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Call an OpenAI-compatible chat-completions endpoint.

    Returns the assistant's reply text, or an empty string on repeated failure.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    url = api_base.rstrip("/") + "/chat/completions"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            logger.warning("Request timed out (attempt %d/%d).", attempt, retries)
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error %s (attempt %d/%d).", exc, attempt, retries)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning(
                "Unexpected response format: %s (attempt %d/%d).", exc, attempt, retries
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Request error: %s (attempt %d/%d).", exc, attempt, retries)

        if attempt < retries:
            time.sleep(retry_delay)

    logger.error("All %d attempts failed; returning empty prediction.", retries)
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_task_data(data_path: str) -> list:
    """Load a task JSON file and return a list of examples."""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Some files may be stored as a dict with integer-string keys
    return [data[str(i)] for i in range(len(data))]


# ---------------------------------------------------------------------------
# Prediction saving
# ---------------------------------------------------------------------------

def save_predictions(predictions: dict, output_path: str) -> None:
    """Write predictions dict to a JSON file, creating parent dirs if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> str:
    """
    Run inference over all task files in *args.data_dir* and save results
    under *args.output_dir*/<model_name>/.

    Returns the path to the model-specific output folder.
    """
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    model_output_dir = Path(args.output_dir) / args.model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Character budget for the prompt (leave room for the generated tokens)
    # We use chars ≈ 2 × tokens as a conservative upper bound for Chinese.
    max_prompt_chars = (args.max_tokens - args.max_new_tokens) * 2

    task_files = sorted(data_dir.glob("*.json"))
    if not task_files:
        logger.error("No JSON task files found in %s", data_dir)
        sys.exit(1)

    # Optionally restrict to specific tasks
    selected_tasks = set(args.tasks) if args.tasks else None

    for task_file in task_files:
        task_id = task_file.stem  # e.g. "1-1"
        if selected_tasks and task_id not in selected_tasks:
            continue

        output_file = model_output_dir / f"{task_id}.json"
        if output_file.exists() and not args.overwrite:
            logger.info("Skipping %s (output already exists; use --overwrite to redo).", task_id)
            continue

        logger.info("Processing task %s …", task_id)
        try:
            examples = load_task_data(str(task_file))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.error("Failed to load %s: %s", task_file, exc)
            continue

        predictions = {}
        for idx, example in enumerate(examples):
            instruction = example.get("instruction", "")
            question = example.get("question", "")
            answer = example.get("answer", "")

            # Build prompt: instruction + newline + question
            prompt = f"{instruction}\n{question}" if instruction else question

            # Truncate if necessary (keep *last* max_prompt_chars)
            prompt = truncate_prompt(prompt, max_prompt_chars)

            prediction_text = call_llm(
                prompt=prompt,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )

            predictions[str(idx)] = {
                "origin_prompt": [{"role": "HUMAN", "prompt": prompt}],
                "prediction": prediction_text,
                "refr": answer,
            }

            if (idx + 1) % 50 == 0:
                logger.info("  … %d / %d examples done", idx + 1, len(examples))

        save_predictions(predictions, str(output_file))
        logger.info("Saved predictions for task %s → %s", task_id, output_file)

    return str(model_output_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run LawBench inference for any OpenAI-compatible LLM."
    )
    # API configuration
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (env: OPENAI_API_BASE)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (env: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="Model identifier to pass in the API request",
    )
    # Data / output
    parser.add_argument(
        "--model-name",
        required=True,
        help="Display name for this model; used as the sub-folder name under output-dir",
    )
    parser.add_argument(
        "--data-dir",
        default="data/zero_shot",
        help="Directory containing the benchmark task JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="predictions/zero_shot",
        help="Root directory where prediction files are written",
    )
    # Token budget
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Total context window of the model (prompt + generation)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    # Generation options
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    # Filtering
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional list of task IDs to run (e.g. 1-1 2-3). Defaults to all tasks.",
    )
    # Retry / resilience
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of API call retries on failure",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between retries",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction files",
    )
    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation/main.py after inference and print the score CSV",
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Path for the evaluation CSV output (default: <output-dir>/results.csv)",
    )

    args = parser.parse_args(argv)

    if not args.api_key:
        logger.warning(
            "No API key provided.  Set --api-key or OPENAI_API_KEY environment variable."
        )

    # Run inference
    model_output_dir = run_inference(args)

    # Optionally run evaluation
    if args.evaluate:
        eval_output = args.eval_output or os.path.join(args.output_dir, "results.csv")
        eval_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "evaluation", "main.py"
        )
        # Use subprocess so that sys.argv is clean for evaluation/main.py.
        # Evaluation errors are logged but do not raise; the partial CSV is
        # still useful even when individual tasks fail.
        import subprocess

        logger.info("Running evaluation …")
        result = subprocess.run(
            [
                sys.executable,
                eval_script,
                "-i",
                args.output_dir,
                "-o",
                eval_output,
            ],
        )
        if result.returncode != 0:
            logger.error("Evaluation script exited with code %d.", result.returncode)
        else:
            logger.info("Evaluation complete.  Results saved to %s", eval_output)


if __name__ == "__main__":
    main()
