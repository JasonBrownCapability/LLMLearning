"""Re-run evaluation on saved models without retraining.

Usage:
    # Evaluate condition B (LoRA) on all benchmarks
    python -m experiment.eval_only --condition b --output-dir /workspace/results

    # Evaluate condition C on GSM8K-Hard only
    python -m experiment.eval_only --condition c --output-dir /workspace/results --benchmarks gsm8k-hard

    # Evaluate on specific benchmarks
    python -m experiment.eval_only --condition b --output-dir /workspace/results --benchmarks gsm8k,gsm8k-hard
"""

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel

from .config import ExperimentConfig
from .data import load_gsm8k_test, load_gsm8k_hard_test, load_math_test
from .evaluate import evaluate_gsm8k, evaluate_math
from .model_surgery import (
    freeze_base_model,
    insert_layers,
    load_base_model,
    unfreeze_inserted_layers,
)


def main():
    parser = argparse.ArgumentParser(
        description="Re-run evaluation on saved models"
    )
    parser.add_argument(
        "--condition", type=str, required=True,
        choices=["a", "b", "c"],
        help="Which condition to evaluate",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Base directory for saving evaluation results",
    )
    parser.add_argument(
        "--weights-dir", type=str, default=None,
        help="Directory containing saved weights (default: same as --output-dir)",
    )
    parser.add_argument(
        "--pass-at-k", type=int, default=1,
        help="Number of sampled generations for pass@k metric",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to this many samples per benchmark",
    )
    parser.add_argument(
        "--benchmarks", type=str, default="gsm8k,gsm8k-hard,math",
        help="Comma-separated benchmarks to run (default: gsm8k,gsm8k-hard,math)",
    )
    args = parser.parse_args()
    args.benchmarks = [b.strip() for b in args.benchmarks.split(",")]
    weights_dir = args.weights_dir or args.output_dir

    config = ExperimentConfig()
    config.output_dir = args.output_dir

    if args.condition == "a":
        # Baseline — just load and evaluate
        model, tokenizer = load_base_model(config.model)
        condition_name = "condition_a_baseline"

    elif args.condition == "b":
        # LoRA — load base model + saved adapter
        model, tokenizer = load_base_model(config.model)
        adapter_dir = os.path.join(weights_dir, "condition_b_lora_rl")
        print(f"Loading LoRA adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        condition_name = "condition_b_lora_rl"

    elif args.condition == "c":
        # Inserted layers — load base model, insert layers, load saved weights
        model, tokenizer = load_base_model(config.model)
        freeze_base_model(model)
        inserted_indices = insert_layers(model, config.inserted_layers)
        unfreeze_inserted_layers(model, inserted_indices)

        weights_path = os.path.join(
            weights_dir, "condition_c_inserted_rl", "inserted_layers.pt"
        )
        print(f"Loading inserted layer weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        layers = model.model.layers
        for idx in inserted_indices:
            prefix = f"inserted_layer_{idx}."
            layer_state = {
                k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            layers[idx].load_state_dict(layer_state)
            # Move to correct device/dtype
            ref_param = next(layers[0].parameters())
            dtype = ref_param.dtype if ref_param.dtype.is_floating_point else torch.bfloat16
            layers[idx] = layers[idx].to(device=ref_param.device, dtype=dtype)

        condition_name = "condition_c_inserted_rl"

    output_path = Path(args.output_dir) / (condition_name + "_reeval")
    output_path.mkdir(parents=True, exist_ok=True)
    results = {"condition": condition_name}

    if "gsm8k" in args.benchmarks:
        print("\nRunning GSM8K evaluation...")
        gsm8k_results = evaluate_gsm8k(
            model, tokenizer,
            max_samples=args.max_samples,
            num_samples_pass_at_k=args.pass_at_k,
        )
        print(f"GSM8K pass@1: {gsm8k_results['accuracy_at_1']:.4f}")
        with open(output_path / "gsm8k_results.json", "w") as f:
            json.dump(gsm8k_results, f, indent=2, default=str)
        results["gsm8k_pass_at_1"] = gsm8k_results["accuracy_at_1"]

    if "gsm8k-hard" in args.benchmarks:
        print("\nRunning GSM8K-Hard evaluation...")
        gsm8k_hard_results = evaluate_gsm8k(
            model, tokenizer,
            max_samples=args.max_samples,
            num_samples_pass_at_k=1,
            dataset_override=load_gsm8k_hard_test(),
        )
        print(f"GSM8K-Hard pass@1: {gsm8k_hard_results['accuracy_at_1']:.4f}")
        with open(output_path / "gsm8k_hard_results.json", "w") as f:
            json.dump(gsm8k_hard_results, f, indent=2, default=str)
        results["gsm8k_hard_pass_at_1"] = gsm8k_hard_results["accuracy_at_1"]

    if "math" in args.benchmarks:
        print("\nRunning MATH evaluation...")
        math_results = evaluate_math(
            model, tokenizer,
            max_samples=args.max_samples,
        )
        print(f"MATH accuracy: {math_results['accuracy']:.4f}")
        with open(output_path / "math_results.json", "w") as f:
            json.dump(math_results, f, indent=2, default=str)
        results["math_accuracy"] = math_results["accuracy"]

    with open(output_path / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
