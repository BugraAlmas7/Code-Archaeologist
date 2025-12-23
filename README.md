# LiveCodeBench Evaluation Pipeline

Author: naholav

A comprehensive evaluation pipeline for LoRA fine-tuned code generation models using the LiveCodeBench benchmark.

**Note:** All models in this project were evaluated on **AtCoder Easy problems (41 questions)**. We recommend using the same evaluation setup for consistent and comparable results.

## Overview

This evaluation pipeline tests LoRA fine-tuned models on LiveCodeBench, a benchmark containing competitive programming problems from platforms like AtCoder, LeetCode, and Codeforces.

## Directory Structure

```
test_github/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── livecodebench_eval.py       # Main evaluation script
├── run_all_evaluations.py      # Batch evaluation runner
└── common/
    ├── __init__.py
    ├── model_loader.py         # Model loading utilities
    ├── code_executor.py        # Safe code execution
    └── code_postprocess.py     # Output post-processing
```

## Requirements

- Python 3.9+
- CUDA-compatible GPU with at least 8GB VRAM
- Linux (recommended) or macOS

## Installation

1. Clone this repository and navigate to the evaluation directory:

```bash
cd test_github
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Install Flash Attention 2 for faster inference:

```bash
pip install flash-attn --no-build-isolation
```

## Checkpoint Directory Structure

Your trained model checkpoints must follow this directory structure:

```
models/
├── deep_think/
│   └── checkpoints/
│       ├── checkpoint-step-300-epoch-2/
│       ├── checkpoint-step-400-epoch-2/
│       ├── checkpoint-step-500-epoch-2/
│       └── ...
├── deep_instruction/
│   └── checkpoints/
│       └── ...
├── diverse_think/
│   └── checkpoints/
│       └── ...
└── diverse_instruction/
    └── checkpoints/
        └── ...
```

**Important:** If your model types have different names, update the `model_types` tuple in the configuration section of the scripts.

## Configuration

### Modifying Model Types

If your training setup uses different model type names, edit the `Config` class in `livecodebench_eval.py`:

```python
# Default configuration
model_types: tuple = (
    "deep_think",
    "deep_instruction",
    "diverse_think",
    "diverse_instruction"
)

# Example: Custom model types
model_types: tuple = (
    "my_model_type_1",
    "my_model_type_2",
)
```

### Modifying Checkpoint Steps

If your checkpoints are saved at different steps:

```python
# Default
checkpoint_steps: tuple = (300, 400, 500, 600, 700, 800)

# Custom steps
checkpoint_steps: tuple = (100, 200, 300, 400, 500)
```

### Modifying System Prompts

The system prompts must match what was used during training. Edit in `Config.__post_init__()`:

```python
self.system_prompts = {
    "think": "Your think-style system prompt here",
    "instruction": "Your instruction-style system prompt here"
}
```

## Usage

### Single Model Evaluation

Evaluate a specific model type and checkpoint:

```bash
python livecodebench_eval.py \
    --model_type deep_think \
    --steps 600 \
    --difficulty easy \
    --checkpoint_dir ./models \
    --output_dir ./results
```

### Evaluate All Models on Easy Difficulty

```bash
python livecodebench_eval.py \
    --difficulty easy \
    --include_base \
    --checkpoint_dir ./models
```

### Platform-Specific Evaluation

Evaluate only on AtCoder problems:

```bash
python livecodebench_eval.py \
    --platform atcoder \
    --difficulty easy \
    --include_base \
    --checkpoint_dir ./models
```

### Recommended: AtCoder Easy Evaluation

We recommend using AtCoder Easy problems (41 questions) for evaluation because:
- Full LiveCodeBench evaluation takes many hours to complete
- The 1.5B parameter model cannot solve hard problems even with fine-tuning
- AtCoder Easy provides a good balance of speed and meaningful comparison

```bash
python livecodebench_eval.py \
    --platform atcoder \
    --difficulty easy \
    --include_base \
    --checkpoint_dir ./models
```

### Batch Evaluation (All Combinations)

Run comprehensive evaluation across all models and difficulties:

```bash
python run_all_evaluations.py \
    --models_dir ./models \
    --output_dir ./results
```

Quick test mode (single model, single difficulty):

```bash
python run_all_evaluations.py --quick
```

Resume from previous run:

```bash
python run_all_evaluations.py --resume
```

## Command Line Arguments

### livecodebench_eval.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_type` | Model type to evaluate (or "all") | all |
| `--steps` | Checkpoint steps to evaluate | all |
| `--include_base` | Include base model in evaluation | False |
| `--difficulty` | Difficulty filter (easy/medium/hard/all) | all |
| `--platform` | Platform filter (atcoder/leetcode/codeforces/all) | all |
| `--date_start` | Start date in YYMM format | 2408 |
| `--date_end` | End date in YYMM format | 2502 |
| `--checkpoint_dir` | Directory containing model checkpoints | ./models |
| `--output_dir` | Output directory for results | ./results/livecodebench |
| `--version` | LiveCodeBench version | release_v5 |

### run_all_evaluations.py

| Argument | Description | Default |
|----------|-------------|---------|
| `--quick` | Quick test mode (1 model, 1 difficulty) | False |
| `--model_types` | Specific model types to evaluate | all |
| `--steps` | Specific steps to evaluate | all |
| `--difficulties` | Specific difficulties to evaluate | easy, medium, hard |
| `--resume` | Skip existing results | False |
| `--include_base` | Include base model | False |
| `--models_dir` | Checkpoint directory | ./models |
| `--output_dir` | Output directory | ./results/livecodebench |

## Output Files

After evaluation, results are saved in the following structure:

```
results/livecodebench/
├── summary.json                           # Overall summary
├── detailed/
│   └── {model_name}_{difficulty}.jsonl    # Detailed per-problem logs
├── generations/
│   └── {model_name}_{difficulty}.json     # Generated code
└── evaluations/
    └── {model_name}_{difficulty}_results.json  # Evaluation results
```

### Results Format

The summary JSON contains:

```json
{
    "model_name": "deep_think_checkpoint-step-600-epoch-3",
    "pass_at_1": 0.3171,
    "stats": {
        "total": 41,
        "passed": 13,
        "failed": 28,
        "error": 0,
        "no_tests": 0
    }
}
```

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors, try:

1. Use a smaller batch size (already set to 1 by default)
2. Reduce `max_new_tokens` in the config
3. Use a GPU with more VRAM

### Checkpoint Not Found

Ensure your checkpoint directory structure matches the expected format:

```
{checkpoint_dir}/{model_type}/checkpoints/checkpoint-step-{step}-epoch-{epoch}/
```

### Different Model Types

If you used different model type names during training, update the `model_types` tuple in both `livecodebench_eval.py` and `run_all_evaluations.py`.

### Flash Attention Error

If Flash Attention 2 fails to load, the scripts will still work but may be slower. To explicitly disable:

```python
use_flash_attention_2=False
```

## License

MIT License
