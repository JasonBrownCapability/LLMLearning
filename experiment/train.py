"""Main training script for all experiment conditions.

Usage:
    python -m experiment.train --condition a    # Baseline evaluation only
    python -m experiment.train --condition b    # LoRA + GRPO
    python -m experiment.train --condition c    # Inserted layers + GRPO
    python -m experiment.train --condition d    # Inserted layers + SFT
    python -m experiment.train --condition e    # Two-stage (LoRA then inserted layers)

    # Quick test (10 steps, 50 eval samples):
    python -m experiment.train --condition b --test-run

    # Local CPU smoke test (tiny model, no GPU required):
    python -m experiment.train --condition a --smoke-test
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer, SFTConfig as TRLSFTConfig, SFTTrainer

from .config import ExperimentConfig
from .data import load_gsm8k_train, load_gsm8k_sft
from .evaluate import run_full_evaluation
from .model_surgery import (
    freeze_base_model,
    insert_layers,
    load_base_model,
    print_model_summary,
    unfreeze_inserted_layers,
    verify_noop,
)
from .rewards import gsm8k_reward_fn


def _set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GradientNormCallback(TrainerCallback):
    """Logs gradient norms split by layer group (inserted vs base)."""

    def __init__(self, model, inserted_indices):
        self.model = model
        self.inserted_indices = set(inserted_indices)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return

        inserted_grad_norms = []
        base_grad_norms = []

        for idx, layer in enumerate(self.model.model.layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if idx in self.inserted_indices:
                        inserted_grad_norms.append(grad_norm)
                    else:
                        base_grad_norms.append(grad_norm)

        logs = {}
        if inserted_grad_norms:
            logs["grad_norm/inserted_mean"] = sum(inserted_grad_norms) / len(inserted_grad_norms)
            logs["grad_norm/inserted_max"] = max(inserted_grad_norms)
        if base_grad_norms:
            logs["grad_norm/base_mean"] = sum(base_grad_norms) / len(base_grad_norms)
            logs["grad_norm/base_max"] = max(base_grad_norms)

        if logs:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(logs, step=state.global_step)
            except ImportError:
                pass


def run_condition_a(config: ExperimentConfig, test_run: bool = False, smoke_test: bool = False, pass_at_k: int = 1, max_eval_samples: int = None):
    """Condition A: Baseline — evaluate the unmodified base model."""
    print("\n" + "=" * 60)
    print("CONDITION A: Baseline (no training)")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model, smoke_test=smoke_test)
    print_model_summary(model)

    max_samples = 50 if test_run else max_eval_samples
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_a_baseline",
        max_samples=max_samples,
        num_samples_pass_at_k=pass_at_k,
    )
    return results


def run_condition_b(config: ExperimentConfig, test_run: bool = False, smoke_test: bool = False, pass_at_k: int = 1, max_eval_samples: int = None):
    """Condition B: LoRA + GRPO on frozen base."""
    print("\n" + "=" * 60)
    print("CONDITION B: LoRA + GRPO")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model, smoke_test=smoke_test)

    # Configure LoRA (passed directly to GRPOTrainer via peft_config)
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load training data
    train_dataset = load_gsm8k_train()

    # Configure GRPO
    max_steps = 10 if test_run else config.grpo.max_steps
    output_dir = os.path.join(config.output_dir, "condition_b_lora_rl")

    grpo_config = TRLGRPOConfig(
        output_dir=output_dir,
        num_generations=config.grpo.num_rollouts,
        learning_rate=config.grpo.learning_rate,
        warmup_steps=config.grpo.warmup_steps,
        max_steps=max_steps,
        per_device_train_batch_size=config.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        gradient_checkpointing=config.grpo.gradient_checkpointing,
        beta=config.grpo.kl_coef,
        temperature=config.grpo.temperature,
        max_completion_length=config.grpo.max_completion_length,
        logging_steps=config.grpo.logging_steps,
        save_steps=config.grpo.save_steps if not test_run else max_steps,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name="condition_b_lora_rl",
        bf16=not smoke_test,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nStarting LoRA + GRPO training...")
    trainer.train()
    trainer.save_model(output_dir)

    # Use the trainer's model (which has LoRA applied) for evaluation
    model = trainer.model

    # Evaluate
    max_samples = 50 if test_run else max_eval_samples
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_b_lora_rl",
        max_samples=max_samples,
        num_samples_pass_at_k=pass_at_k,
    )
    return results


def run_condition_c(config: ExperimentConfig, test_run: bool = False, smoke_test: bool = False, pass_at_k: int = 1, max_eval_samples: int = None):
    """Condition C: Inserted layers + GRPO on frozen base."""
    print("\n" + "=" * 60)
    print("CONDITION C: Inserted Layers + GRPO")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model, smoke_test=smoke_test)

    # Insert layers and configure trainability
    freeze_base_model(model)
    inserted_indices = insert_layers(model, config.inserted_layers)
    unfreeze_inserted_layers(model, inserted_indices)

    # Verify no-op before training
    is_noop = verify_noop(model, tokenizer, inserted_indices)
    if not is_noop and config.inserted_layers.init_strategy == "zero":
        print("ERROR: Inserted layers are not no-ops. Aborting.")
        sys.exit(1)

    print_model_summary(model)

    # Load training data
    train_dataset = load_gsm8k_train()

    # Configure GRPO
    max_steps = 10 if test_run else config.grpo.max_steps
    output_dir = os.path.join(config.output_dir, "condition_c_inserted_rl")

    # Lower learning rate for inserted layers — all gradient is concentrated
    # in 2 layers instead of distributed across 32 LoRA adapters
    inserted_lr = config.grpo.learning_rate / 10  # 1e-6 vs 1e-5

    grpo_config = TRLGRPOConfig(
        output_dir=output_dir,
        num_generations=config.grpo.num_rollouts,
        learning_rate=inserted_lr,
        warmup_steps=config.grpo.warmup_steps,
        max_steps=max_steps,
        per_device_train_batch_size=config.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        gradient_checkpointing=config.grpo.gradient_checkpointing,
        beta=config.grpo.kl_coef,
        temperature=config.grpo.temperature,
        max_completion_length=config.grpo.max_completion_length,
        logging_steps=config.grpo.logging_steps,
        save_steps=max_steps + 1,  # Disable auto-save (crashes with _hf_peft_config_loaded hack)
        save_strategy="no",
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name="condition_c_inserted_rl",
        bf16=not smoke_test,
    )

    # Tell the trainer this quantized model has trainable components
    # (inserted layers are non-quantized and trainable, but aren't PEFT adapters)
    model._hf_peft_config_loaded = True

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[GradientNormCallback(model, inserted_indices)],
    )

    print("\nStarting Inserted Layers + GRPO training...")
    trainer.train()

    # Save only the inserted layer weights
    _save_inserted_layers(model, inserted_indices, output_dir)

    # Evaluate
    max_samples = 50 if test_run else max_eval_samples
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_c_inserted_rl",
        max_samples=max_samples,
        num_samples_pass_at_k=pass_at_k,
    )
    return results


def run_condition_d(config: ExperimentConfig, test_run: bool = False, smoke_test: bool = False, pass_at_k: int = 1, max_eval_samples: int = None):
    """Condition D: Inserted layers + SFT on frozen base."""
    print("\n" + "=" * 60)
    print("CONDITION D: Inserted Layers + SFT")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model, smoke_test=smoke_test)

    # Insert layers and configure trainability
    freeze_base_model(model)
    inserted_indices = insert_layers(model, config.inserted_layers)
    unfreeze_inserted_layers(model, inserted_indices)
    verify_noop(model, tokenizer, inserted_indices)
    print_model_summary(model)

    # Load SFT data (includes full solution traces)
    train_dataset = load_gsm8k_sft()

    # Configure SFT
    max_steps = 10 if test_run else config.sft.max_steps
    output_dir = os.path.join(config.output_dir, "condition_d_inserted_sft")

    inserted_lr = config.sft.learning_rate / 10

    sft_config = TRLSFTConfig(
        output_dir=output_dir,
        learning_rate=inserted_lr,
        warmup_steps=config.sft.warmup_steps,
        max_steps=max_steps,
        per_device_train_batch_size=config.sft.per_device_train_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        gradient_checkpointing=config.sft.gradient_checkpointing,
        logging_steps=config.sft.logging_steps,
        save_steps=max_steps + 1,
        save_strategy="no",
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name="condition_d_inserted_sft",
        bf16=not smoke_test,
    )

    # Tell the trainer this quantized model has trainable components
    model._hf_peft_config_loaded = True

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[GradientNormCallback(model, inserted_indices)],
    )

    print("\nStarting Inserted Layers + SFT training...")
    trainer.train()

    _save_inserted_layers(model, inserted_indices, output_dir)

    # Evaluate
    max_samples = 50 if test_run else max_eval_samples
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_d_inserted_sft",
        max_samples=max_samples,
        num_samples_pass_at_k=pass_at_k,
    )
    return results


def run_condition_e(config: ExperimentConfig, test_run: bool = False, smoke_test: bool = False, pass_at_k: int = 1, max_eval_samples: int = None, reuse_lora: str = None):
    """Condition E: Two-stage training — LoRA first, then inserted layers.

    Stage 1: Train LoRA adapters via GRPO to acquire reasoning capability.
             (Skipped if --reuse-lora is provided.)
    Stage 2: Merge LoRA into base weights, insert new layers, and train
             the inserted layers via GRPO on the strengthened base model.
    Stage 3: Evaluate the final model (merged base + trained inserted layers).
    """
    print("\n" + "=" * 60)
    print("CONDITION E: Two-Stage (LoRA then Inserted Layers)")
    print("=" * 60)

    two_stage = config.two_stage
    output_dir = os.path.join(config.output_dir, "condition_e_two_stage")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_base_model(config.model, smoke_test=smoke_test)
    train_dataset = load_gsm8k_train()
    max_samples = 50 if test_run else max_eval_samples

    if reuse_lora:
        # ──────────────────────────────────────────────
        # Stage 1 (skipped): Load pre-trained LoRA
        # ──────────────────────────────────────────────
        print(f"\n--- Stage 1: Loading pre-trained LoRA from {reuse_lora} ---")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, reuse_lora)
        print("LoRA adapter loaded. Merging into base model...")
        model = model.merge_and_unload()
    else:
        # ──────────────────────────────────────────────
        # Stage 1: LoRA + GRPO (fast acquisition)
        # ──────────────────────────────────────────────
        print("\n--- Stage 1: LoRA + GRPO ---")

        lora_config = LoraConfig(
            r=two_stage.lora_rank,
            lora_alpha=two_stage.lora_alpha,
            lora_dropout=0.05,
            target_modules=config.lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        stage1_steps = 10 if test_run else two_stage.stage1_max_steps

        grpo_config = TRLGRPOConfig(
            output_dir=os.path.join(output_dir, "stage1_lora"),
            num_generations=config.grpo.num_rollouts,
            learning_rate=config.grpo.learning_rate,
            warmup_steps=config.grpo.warmup_steps,
            max_steps=stage1_steps,
            per_device_train_batch_size=config.grpo.per_device_train_batch_size,
            gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
            gradient_checkpointing=config.grpo.gradient_checkpointing,
            beta=config.grpo.kl_coef,
            temperature=config.grpo.temperature,
            max_completion_length=config.grpo.max_completion_length,
            logging_steps=config.grpo.logging_steps,
            save_steps=stage1_steps,
            seed=config.seed,
            report_to="wandb" if config.use_wandb else "none",
            run_name="condition_e_stage1_lora",
            bf16=not smoke_test,
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=gsm8k_reward_fn,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        print("Starting Stage 1 training...")
        trainer.train()

        # The trainer wraps the model with PEFT internally. Get the trained model.
        model = trainer.model

        # Evaluate with LoRA (to see what stage 1 achieved)
        print("\nEvaluating after Stage 1 (with LoRA)...")
        run_full_evaluation(
            model, tokenizer,
            output_dir=config.output_dir,
            condition_name="condition_e_stage1_with_lora",
            max_samples=max_samples,
            num_samples_pass_at_k=pass_at_k,
        )

        # Merge LoRA weights into the base model permanently
        model = model.merge_and_unload()

    # ──────────────────────────────────────────────
    # Stage 2: Train inserted layers on LoRA-merged base via SFT
    # ──────────────────────────────────────────────
    print("\n--- Stage 2: Inserted Layers + SFT on LoRA-merged base ---")

    # Now insert layers into the LoRA-merged model
    freeze_base_model(model)
    inserted_indices = insert_layers(model, config.inserted_layers)
    unfreeze_inserted_layers(model, inserted_indices)
    verify_noop(model, tokenizer, inserted_indices)
    print_model_summary(model)

    # Train inserted layers via SFT
    # The model now has LoRA knowledge baked in, plus fresh inserted layers.
    # SFT outperforms GRPO for inserted layers (condition D > C) and trains ~50x faster.
    stage2_steps = 10 if test_run else two_stage.stage2_max_steps

    inserted_lr = config.sft.learning_rate / 10

    sft_config_s2 = TRLSFTConfig(
        output_dir=os.path.join(output_dir, "stage2_inserted"),
        learning_rate=inserted_lr,
        warmup_steps=50,
        max_steps=stage2_steps,
        per_device_train_batch_size=config.sft.per_device_train_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        gradient_checkpointing=config.sft.gradient_checkpointing,
        logging_steps=config.sft.logging_steps,
        save_steps=stage2_steps + 1,
        save_strategy="no",
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name="condition_e_stage2_inserted_sft",
        bf16=not smoke_test,
    )

    # Tell the trainer this quantized model has trainable components
    model._hf_peft_config_loaded = True

    sft_dataset = load_gsm8k_sft()

    trainer_s2 = SFTTrainer(
        model=model,
        args=sft_config_s2,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        callbacks=[GradientNormCallback(model, inserted_indices)],
    )

    print("Starting Stage 2 training (inserted layers + SFT)...")
    trainer_s2.train()

    _save_inserted_layers(model, inserted_indices, output_dir)

    # ──────────────────────────────────────────────
    # Stage 3: Final evaluation
    # ──────────────────────────────────────────────
    print("\n--- Stage 3: Final evaluation ---")

    # The model now has: original base (with LoRA merged) + trained inserted layers.
    # This is the final model — evaluate it.
    print("\nEvaluating final two-stage model...")
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_e_two_stage_final",
        max_samples=max_samples,
        num_samples_pass_at_k=pass_at_k,
    )
    return results


def _save_inserted_layers(model, inserted_indices, output_dir):
    """Save only the inserted layer weights (not the full model)."""
    os.makedirs(output_dir, exist_ok=True)
    state_dict = {}
    for idx in inserted_indices:
        layer = model.model.layers[idx]
        for name, param in layer.named_parameters():
            state_dict[f"inserted_layer_{idx}.{name}"] = param.cpu()

    save_path = os.path.join(output_dir, "inserted_layers.pt")
    torch.save(state_dict, save_path)
    print(f"Saved inserted layer weights to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment conditions for RL through inserted layers"
    )
    parser.add_argument(
        "--condition", type=str, required=True,
        choices=["a", "b", "c", "d", "e"],
        help="Which condition to run (a=baseline, b=lora+rl, c=inserted+rl, "
             "d=inserted+sft, e=two-stage)",
    )
    parser.add_argument(
        "--test-run", action="store_true",
        help="Quick test: 10 training steps, 50 eval samples",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Local CPU smoke test: tiny model, no GPU required (implies --test-run)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (used when --seeds is not provided)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for multi-run aggregation (e.g., --seeds 42,123,456)",
    )
    parser.add_argument(
        "--pass-at-k", type=int, default=1,
        help="Number of sampled generations for pass@k metric (default: 1, use 8 for pass@8)",
    )
    parser.add_argument(
        "--reuse-lora", type=str, default=None,
        help="For condition E: skip stage 1 and load LoRA from this path (e.g., /workspace/results/condition_b_lora_rl)",
    )
    parser.add_argument(
        "--max-eval-samples", type=int, default=None,
        help="Limit evaluation to this many samples per benchmark (default: all)",
    )
    args = parser.parse_args()

    # --smoke-test implies --test-run
    if args.smoke_test:
        args.test_run = True

    config = ExperimentConfig()
    config.output_dir = args.output_dir
    config.use_wandb = not args.no_wandb

    if args.smoke_test:
        config.model.name = "HuggingFaceTB/SmolLM-135M"
        config.model.quantize_4bit = False
        config.use_wandb = False
        print("SMOKE TEST: using SmolLM-135M on CPU (no GPU required)")

    seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    conditions = {
        "a": run_condition_a,
        "b": run_condition_b,
        "c": run_condition_c,
        "d": run_condition_d,
        "e": run_condition_e,
    }

    condition_fn = conditions[args.condition]
    all_results = []

    for seed in seeds:
        config.seed = seed
        _set_seed(seed)
        if len(seeds) > 1:
            print(f"\n{'#'*60}")
            print(f"Running with seed={seed}")
            print(f"{'#'*60}")
        results = condition_fn(config, test_run=args.test_run, smoke_test=args.smoke_test, pass_at_k=args.pass_at_k,
                               max_eval_samples=args.max_eval_samples,
                               **({"reuse_lora": args.reuse_lora} if args.condition == "e" else {}))
        all_results.append(results)

    if len(seeds) > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS for condition {args.condition.upper()} "
              f"across {len(seeds)} seeds:")
        print(f"{'='*60}")
        numeric_keys = [k for k in all_results[0]
                        if isinstance(all_results[0][k], (int, float))]
        for key in numeric_keys:
            values = [r[key] for r in all_results]
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {key}: {mean:.4f} +/- {std:.4f}")
    else:
        print(f"\n{'='*60}")
        print(f"RESULTS for condition {args.condition.upper()}:")
        print(f"{'='*60}")
        for key, value in all_results[0].items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
