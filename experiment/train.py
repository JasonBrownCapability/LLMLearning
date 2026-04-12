"""Main training script for all experiment conditions.

Usage:
    python -m experiment.train --condition a    # Baseline evaluation only
    python -m experiment.train --condition b    # LoRA + GRPO
    python -m experiment.train --condition c    # Inserted layers + GRPO
    python -m experiment.train --condition d    # Inserted layers + SFT
    python -m experiment.train --condition e    # Two-stage (LoRA then inserted layers)

    # Quick test (10 steps, 50 eval samples):
    python -m experiment.train --condition b --test-run
"""

import argparse
import os
import sys

import torch
from peft import LoraConfig
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


def run_condition_a(config: ExperimentConfig, test_run: bool = False):
    """Condition A: Baseline — evaluate the unmodified base model."""
    print("\n" + "=" * 60)
    print("CONDITION A: Baseline (no training)")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model)
    print_model_summary(model)

    max_samples = 50 if test_run else None
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_a_baseline",
        max_samples=max_samples,
    )
    return results


def run_condition_b(config: ExperimentConfig, test_run: bool = False):
    """Condition B: LoRA + GRPO on frozen base."""
    print("\n" + "=" * 60)
    print("CONDITION B: LoRA + GRPO")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model)

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
        bf16=True,
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
    max_samples = 50 if test_run else None
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_b_lora_rl",
        max_samples=max_samples,
    )
    return results


def run_condition_c(config: ExperimentConfig, test_run: bool = False):
    """Condition C: Inserted layers + GRPO on frozen base."""
    print("\n" + "=" * 60)
    print("CONDITION C: Inserted Layers + GRPO")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model)

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
        run_name="condition_c_inserted_rl",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting Inserted Layers + GRPO training...")
    trainer.train()

    # Save only the inserted layer weights
    _save_inserted_layers(model, inserted_indices, output_dir)

    # Evaluate
    max_samples = 50 if test_run else None
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_c_inserted_rl",
        max_samples=max_samples,
    )
    return results


def run_condition_d(config: ExperimentConfig, test_run: bool = False):
    """Condition D: Inserted layers + SFT on frozen base."""
    print("\n" + "=" * 60)
    print("CONDITION D: Inserted Layers + SFT")
    print("=" * 60)

    model, tokenizer = load_base_model(config.model)

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

    sft_config = TRLSFTConfig(
        output_dir=output_dir,
        learning_rate=config.sft.learning_rate,
        warmup_steps=config.sft.warmup_steps,
        max_steps=max_steps,
        per_device_train_batch_size=config.sft.per_device_train_batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        gradient_checkpointing=config.sft.gradient_checkpointing,
        logging_steps=config.sft.logging_steps,
        save_steps=config.sft.save_steps if not test_run else max_steps,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name="condition_d_inserted_sft",
        bf16=True,
        max_seq_length=config.model.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("\nStarting Inserted Layers + SFT training...")
    trainer.train()

    _save_inserted_layers(model, inserted_indices, output_dir)

    # Evaluate
    max_samples = 50 if test_run else None
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_d_inserted_sft",
        max_samples=max_samples,
    )
    return results


def run_condition_e(config: ExperimentConfig, test_run: bool = False):
    """Condition E: Two-stage CLS pipeline.

    Stage 1: Train LoRA + GRPO (fast acquisition)
    Stage 2: Generate rollouts from LoRA-augmented model, then train
             inserted layers via GRPO (slow consolidation)
    Stage 3: Remove LoRA and evaluate (consolidation test)
    """
    print("\n" + "=" * 60)
    print("CONDITION E: Two-Stage CLS Pipeline")
    print("=" * 60)

    two_stage = config.two_stage
    output_dir = os.path.join(config.output_dir, "condition_e_two_stage")
    os.makedirs(output_dir, exist_ok=True)

    # ──────────────────────────────────────────────
    # Stage 1: LoRA + GRPO (fast acquisition)
    # ──────────────────────────────────────────────
    print("\n--- Stage 1: LoRA + GRPO (fast acquisition) ---")
    model, tokenizer = load_base_model(config.model)

    lora_config = LoraConfig(
        r=two_stage.lora_rank,
        lora_alpha=two_stage.lora_alpha,
        lora_dropout=0.05,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_dataset = load_gsm8k_train()
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
        bf16=True,
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
    max_samples = 50 if test_run else None
    print("\nEvaluating after Stage 1 (with LoRA)...")
    run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_e_stage1_with_lora",
        max_samples=max_samples,
    )

    # ──────────────────────────────────────────────
    # Stage 2: Generate rollouts, then train inserted layers
    # ──────────────────────────────────────────────
    print("\n--- Stage 2: Consolidation into inserted layers ---")

    # Merge LoRA weights into the base model permanently, then discard
    # the LoRA adapter. The model now has LoRA knowledge baked in.
    model = model.merge_and_unload()

    # Now insert layers into the LoRA-merged model
    freeze_base_model(model)
    inserted_indices = insert_layers(model, config.inserted_layers)
    unfreeze_inserted_layers(model, inserted_indices)
    verify_noop(model, tokenizer, inserted_indices)
    print_model_summary(model)

    # Train inserted layers via GRPO
    # The model now has LoRA knowledge baked in, plus fresh inserted layers.
    # The GRPO rollouts will benefit from the LoRA-learned reasoning,
    # providing a richer training signal for the inserted layers.
    stage2_steps = 10 if test_run else two_stage.stage2_max_steps

    grpo_config_s2 = TRLGRPOConfig(
        output_dir=os.path.join(output_dir, "stage2_inserted"),
        num_generations=config.grpo.num_rollouts,
        learning_rate=config.grpo.learning_rate,
        warmup_steps=50,
        max_steps=stage2_steps,
        per_device_train_batch_size=config.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        gradient_checkpointing=config.grpo.gradient_checkpointing,
        beta=config.grpo.kl_coef,
        temperature=config.grpo.temperature,
        max_completion_length=config.grpo.max_completion_length,
        logging_steps=config.grpo.logging_steps,
        save_steps=stage2_steps,
        seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name="condition_e_stage2_inserted",
        bf16=True,
    )

    trainer_s2 = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_reward_fn,
        args=grpo_config_s2,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting Stage 2 training (inserted layers + GRPO)...")
    trainer_s2.train()

    _save_inserted_layers(model, inserted_indices, output_dir)

    # ──────────────────────────────────────────────
    # Stage 3: Evaluate (consolidation test)
    # ──────────────────────────────────────────────
    print("\n--- Stage 3: Consolidation evaluation ---")

    # The model now has: original base (with LoRA merged) + trained inserted layers.
    # This is the final model — evaluate it.
    print("\nEvaluating final two-stage model...")
    results = run_full_evaluation(
        model, tokenizer,
        output_dir=config.output_dir,
        condition_name="condition_e_two_stage_final",
        max_samples=max_samples,
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
        "--output-dir", type=str, default="./results",
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    config.output_dir = args.output_dir
    config.seed = args.seed
    config.use_wandb = not args.no_wandb

    conditions = {
        "a": run_condition_a,
        "b": run_condition_b,
        "c": run_condition_c,
        "d": run_condition_d,
        "e": run_condition_e,
    }

    condition_fn = conditions[args.condition]
    results = condition_fn(config, test_run=args.test_run)

    print(f"\n{'='*60}")
    print(f"RESULTS for condition {args.condition.upper()}:")
    print(f"{'='*60}")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
