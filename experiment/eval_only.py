"""Re-run evaluation on saved models without retraining.

Usage:
    # Evaluate condition B (LoRA) — loads saved adapter weights
    python -m experiment.eval_only --condition b --output-dir /workspace/results

    # Evaluate condition C (inserted layers) — loads saved inserted layer weights
    python -m experiment.eval_only --condition c --output-dir /workspace/results

    # Evaluate baseline (no saved weights needed)
    python -m experiment.eval_only --condition a --output-dir /workspace/results
"""

import argparse
import os

import torch
from peft import PeftModel

from .config import ExperimentConfig
from .evaluate import run_full_evaluation
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
        help="Base directory for outputs (same as training)",
    )
    parser.add_argument(
        "--pass-at-k", type=int, default=1,
        help="Number of sampled generations for pass@k metric",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation to this many samples per benchmark",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    config.output_dir = args.output_dir

    if args.condition == "a":
        # Baseline — just load and evaluate
        model, tokenizer = load_base_model(config.model)
        condition_name = "condition_a_baseline"

    elif args.condition == "b":
        # LoRA — load base model + saved adapter
        model, tokenizer = load_base_model(config.model)
        adapter_dir = os.path.join(args.output_dir, "condition_b_lora_rl")
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
            args.output_dir, "condition_c_inserted_rl", "inserted_layers.pt"
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

    print(f"\nEvaluating {condition_name}...")
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=args.output_dir,
        condition_name=condition_name + "_reeval",
        max_samples=args.max_samples,
        num_samples_pass_at_k=args.pass_at_k,
    )

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
