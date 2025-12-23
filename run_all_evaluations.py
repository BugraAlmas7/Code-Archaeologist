#!/usr/bin/env python3
"""
Run All LiveCodeBench Evaluations

Author: naholav

This script runs comprehensive evaluations of all LoRA fine-tuned models
on LiveCodeBench benchmark across all difficulty levels.

Usage:
    # Run all evaluations (will take several hours)
    python run_all_evaluations.py

    # Quick test with one model and one difficulty
    python run_all_evaluations.py --quick

    # Specific model types only
    python run_all_evaluations.py --model_types deep_think diverse_think

    # Specific steps only
    python run_all_evaluations.py --steps 500 800

    # Resume from previous run
    python run_all_evaluations.py --resume
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import subprocess

import torch
from tqdm import tqdm


# =============================================================================
# Configuration - MODIFY THESE PATHS FOR YOUR SETUP
# =============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Base model
    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # =========================================================================
    # IMPORTANT: Update these paths for your setup
    # =========================================================================
    models_dir: str = "./models"                    # <-- YOUR_CHECKPOINT_PATH
    output_dir: str = "./results/livecodebench"     # <-- YOUR_OUTPUT_PATH

    # Model types (must match your training setup)
    model_types: tuple = (
        "deep_think",
        "deep_instruction",
        "diverse_think",
        "diverse_instruction"
    )

    # Checkpoint steps
    steps: tuple = (300, 400, 500, 600, 700, 800)

    # Difficulties
    difficulties: tuple = ("easy", "medium", "hard")

    # LiveCodeBench version
    lcb_version: str = "release_v5"

    # Generation settings
    max_new_tokens: int = 8192
    temperature: float = 0.0

    # Execution settings
    timeout_per_test: float = 10.0
    memory_limit_mb: int = 512


CONFIG = EvalConfig()


# =============================================================================
# Utility Functions
# =============================================================================

def get_checkpoint_path(model_type: str, step: int) -> Optional[str]:
    """Get checkpoint path for a model type and step."""
    base = Path(CONFIG.models_dir) / model_type / "checkpoints"
    pattern = f"checkpoint-step-{step}-epoch-*"

    matches = list(base.glob(pattern))
    if matches:
        return str(matches[0])
    return None


def get_all_checkpoints() -> List[Dict[str, Any]]:
    """Get all available checkpoints."""
    checkpoints = []

    for model_type in CONFIG.model_types:
        for step in CONFIG.steps:
            path = get_checkpoint_path(model_type, step)
            if path:
                checkpoints.append({
                    "model_type": model_type,
                    "step": step,
                    "path": path,
                    "name": f"{model_type}_step{step}"
                })

    return checkpoints


def estimate_runtime(num_models: int, num_problems: int) -> str:
    """Estimate total runtime."""
    total_seconds = num_models * num_problems * 35
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"~{hours}h {minutes}m"


# =============================================================================
# Evaluation Runner
# =============================================================================

def run_single_evaluation(
    checkpoint: Dict[str, Any],
    difficulty: str,
    resume: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation for a single checkpoint on a specific difficulty.
    """
    model_name = checkpoint["name"]
    output_file = Path(CONFIG.output_dir) / "evaluations" / f"{model_name}_{difficulty}_results.json"

    if resume and output_file.exists():
        print(f"  Skipping {model_name} ({difficulty}) - results exist")
        with open(output_file) as f:
            return json.load(f)

    cmd = [
        sys.executable,
        "livecodebench_eval.py",
        "--model_type", checkpoint["model_type"],
        "--difficulty", difficulty,
        "--steps", str(checkpoint["step"]),
        "--version", CONFIG.lcb_version,
        "--output_dir", CONFIG.output_dir,
        "--checkpoint_dir", CONFIG.models_dir
    ]

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per evaluation
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:500]}")
            return {
                "model_name": model_name,
                "difficulty": difficulty,
                "error": result.stderr
            }

        if output_file.exists():
            with open(output_file) as f:
                return json.load(f)

        return {
            "model_name": model_name,
            "difficulty": difficulty,
            "status": "completed",
            "output": result.stdout[-1000:]
        }

    except subprocess.TimeoutExpired:
        return {
            "model_name": model_name,
            "difficulty": difficulty,
            "error": "timeout"
        }
    except Exception as e:
        return {
            "model_name": model_name,
            "difficulty": difficulty,
            "error": str(e)
        }


def create_results_table(results: List[Dict[str, Any]]) -> str:
    """Create a formatted results table."""
    lines = []
    lines.append("\n" + "="*100)
    lines.append("EVALUATION RESULTS SUMMARY")
    lines.append("="*100)

    by_model = {}
    for r in results:
        model = r.get("model_name", "unknown")
        if model not in by_model:
            by_model[model] = {}
        diff = r.get("difficulty", "unknown")
        by_model[model][diff] = r

    lines.append(f"\n{'Model':<40} {'Easy':<15} {'Medium':<15} {'Hard':<15}")
    lines.append("-"*85)

    for model in sorted(by_model.keys()):
        diffs = by_model[model]
        easy = diffs.get("easy", {}).get("pass_at_1", "-")
        medium = diffs.get("medium", {}).get("pass_at_1", "-")
        hard = diffs.get("hard", {}).get("pass_at_1", "-")

        if isinstance(easy, float):
            easy = f"{easy*100:.1f}%"
        if isinstance(medium, float):
            medium = f"{medium*100:.1f}%"
        if isinstance(hard, float):
            hard = f"{hard*100:.1f}%"

        lines.append(f"{model:<40} {easy:<15} {medium:<15} {hard:<15}")

    lines.append("="*100)

    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all LiveCodeBench evaluations")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (one model, one difficulty)"
    )

    parser.add_argument(
        "--model_types",
        nargs="+",
        default=None,
        help="Specific model types to evaluate"
    )

    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Specific steps to evaluate"
    )

    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=None,
        help="Specific difficulties to evaluate"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip existing results)"
    )

    parser.add_argument(
        "--include_base",
        action="store_true",
        help="Include base model evaluation"
    )

    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="Directory containing model checkpoints"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/livecodebench",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Update config with command line args
    CONFIG.models_dir = args.models_dir
    CONFIG.output_dir = args.output_dir

    print("="*80)
    print("LIVECODEBENCH COMPREHENSIVE EVALUATION")
    print("Author: naholav")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base model: {CONFIG.base_model}")
    print(f"Checkpoint directory: {CONFIG.models_dir}")
    print(f"LiveCodeBench version: {CONFIG.lcb_version}")
    print("="*80)

    all_checkpoints = get_all_checkpoints()

    if args.model_types:
        all_checkpoints = [c for c in all_checkpoints if c["model_type"] in args.model_types]
    if args.steps:
        all_checkpoints = [c for c in all_checkpoints if c["step"] in args.steps]

    if args.quick:
        all_checkpoints = all_checkpoints[:1]
        difficulties = ["easy"]
        print("QUICK MODE: Testing with 1 model, 1 difficulty")
    else:
        difficulties = list(args.difficulties or CONFIG.difficulties)

    if args.include_base:
        all_checkpoints.insert(0, {
            "model_type": "base",
            "step": 0,
            "path": None,
            "name": "base_model"
        })

    print(f"\nCheckpoints to evaluate: {len(all_checkpoints)}")
    for ckpt in all_checkpoints:
        print(f"  - {ckpt['name']}")

    print(f"\nDifficulties: {difficulties}")

    total_evals = len(all_checkpoints) * len(difficulties)
    print(f"\nTotal evaluations: {total_evals}")
    print(f"Estimated runtime: {estimate_runtime(len(all_checkpoints), 200)}")

    if not args.quick:
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    Path(CONFIG.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    failed = []

    print("\n" + "="*80)
    print("RUNNING EVALUATIONS")
    print("="*80)

    for ckpt in all_checkpoints:
        print(f"\n[{ckpt['name']}]")

        for difficulty in difficulties:
            print(f"  Evaluating on {difficulty}...")

            result = run_single_evaluation(ckpt, difficulty, resume=args.resume)
            all_results.append(result)

            if "error" in result:
                failed.append(f"{ckpt['name']}_{difficulty}")
                print(f"  FAILED: {result.get('error', 'unknown error')[:100]}")
            else:
                pass_rate = result.get("pass_at_1", "N/A")
                if isinstance(pass_rate, float):
                    print(f"  Pass@1: {pass_rate*100:.1f}%")

            torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results = {
        "timestamp": timestamp,
        "config": {
            "base_model": CONFIG.base_model,
            "lcb_version": CONFIG.lcb_version,
            "model_types": list(CONFIG.model_types),
            "steps": list(CONFIG.steps),
            "difficulties": list(difficulties)
        },
        "results": all_results,
        "failed": failed
    }

    results_file = Path(CONFIG.output_dir) / f"comprehensive_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(create_results_table(all_results))

    print(f"\nResults saved to: {results_file}")
    print(f"Total evaluations: {len(all_results)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed evaluations:")
        for f_name in failed:
            print(f"  - {f_name}")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
