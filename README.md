# RL Through Inserted Transformer Layers — Experiment

This experiment tests whether Reinforcement Learning through newly-inserted transformer layers (adding serial computational depth) produces better reasoning improvements than RL through LoRA (modifying existing computations), when both operate on a frozen base model.

See `RL-Through-Inserted-Layers.md` for the full research document and `Experiment-Design.md` for the experimental design rationale.

## Prerequisites

- **Python** 3.10+
- **CUDA** 11.8+ with a compatible GPU
- **GPU**: Minimum 1x A100 80GB (or 2x A6000 48GB). For test runs, a 24GB GPU may work.
- **HuggingFace account** with access to Llama 3 8B (request access at https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Weights & Biases account** (optional, for logging)

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or
venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Log in to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted. You need this to download Llama 3 8B.

### 4. (Optional) Log in to Weights & Biases

```bash
wandb login
```

Or pass `--no-wandb` to all commands to skip logging.

## Running the Experiment

All conditions are run via the same script. Run from the project root directory (`LLMLearning/`).

### Quick test (verify everything works)

```bash
python -m experiment.train --condition a --test-run --no-wandb
```

This loads the model, runs evaluation on 50 GSM8K examples, and exits. Should complete in a few minutes and confirms your setup is correct.

### Condition A: Baseline

Evaluate the unmodified base model (no training).

```bash
python -m experiment.train --condition a
```

### Condition B: LoRA + GRPO

Train LoRA adapters via GRPO on GSM8K. This is the main baseline to beat.

```bash
python -m experiment.train --condition b
```

Estimated time: 8-12 hours on 1x A100.

### Condition C: Inserted Layers + GRPO

**This is the core experiment.** Insert 2 new transformer layers (small-random-initialised, near-zero effect) and train them via GRPO.

```bash
python -m experiment.train --condition c
```

Estimated time: 10-14 hours on 1x A100.

### Condition D: Inserted Layers + SFT

Control condition: same inserted layers, but trained via supervised fine-tuning instead of RL. This isolates the RL vs SFT contribution.

```bash
python -m experiment.train --condition d
```

Estimated time: 4-6 hours on 1x A100.

### Condition E: Two-Stage (LoRA then Inserted Layers)

Two-stage training: LoRA via GRPO first (to strengthen the base), then merge LoRA into base weights and train inserted layers via GRPO on the improved model.

```bash
python -m experiment.train --condition e
```

Estimated time: 12-16 hours on 1x A100.

## Recommended Run Order

1. **Condition A** first — establishes baseline and confirms setup
2. **Condition B** second — the main comparison baseline (LoRA + RL)
3. **Condition C** third — the core hypothesis test
4. Compare B vs C. If C shows improvement, continue with D and E.
5. **Condition D** — isolates whether RL matters (vs SFT for inserted layers)
6. **Condition E** — tests whether two-stage scaffolding helps

## Output

Results are saved to `./results/` (or the directory specified via `--output-dir`):

```
results/
  condition_a_baseline/
    gsm8k_results.json     # Per-example GSM8K results
    math_results.json      # Per-example MATH results
    summary.json            # Accuracy metrics (all benchmarks)
  condition_b_lora_rl/
    gsm8k_results.json
    math_results.json
    summary.json
    adapter_model/          # Saved LoRA weights
  condition_c_inserted_rl/
    gsm8k_results.json
    math_results.json
    summary.json
    inserted_layers.pt      # Saved inserted layer weights
  ...
```

## Comparing Results

After running conditions A, B, and C, compare summaries:

```bash
cat results/condition_a_baseline/summary.json
cat results/condition_b_lora_rl/summary.json
cat results/condition_c_inserted_rl/summary.json
```

The key comparison is **B vs C** (LoRA+RL vs Inserted Layers+RL) on GSM8K pass@1.

## Multi-Seed Runs

For more robust results, run with multiple seeds and get aggregated statistics:

```bash
python -m experiment.train --condition c --seeds 42,123,456
```

This runs the condition once per seed and reports mean ± std across all runs.

## Configuration

All hyperparameters are in `experiment/config.py`. Key settings:

| Parameter | Default | Notes |
|---|---|---|
| `InsertedLayerConfig.positions` | [10, 21] | Where to insert layers (0-indexed) |
| `InsertedLayerConfig.num_layers` | 2 | Number of layers to insert |
| `InsertedLayerConfig.init_strategy` | "small_random" | Near-zero init for gradient flow ("zero" available for ablation) |
| `LoRAConfig.rank` | 64 | LoRA rank (~500M params to match inserted layers) |
| `GRPOConfig.max_steps` | 2000 | Training steps (~1 epoch on GSM8K) |
| `GRPOConfig.num_rollouts` | 8 | Completions per prompt for GRPO |
| `GRPOConfig.kl_coef` | 0.01 | KL penalty coefficient |

## Troubleshooting

### Out of memory

- Reduce `GRPOConfig.num_rollouts` from 8 to 4
- Reduce `GRPOConfig.per_device_train_batch_size` from 4 to 2
- Increase `GRPOConfig.gradient_accumulation_steps` to compensate
- Ensure `gradient_checkpointing` is True (default)

### Model download fails

- Confirm you have accepted the Llama 3 licence at https://huggingface.co/meta-llama/Meta-Llama-3-8B
- Confirm `huggingface-cli login` succeeded
- Try setting `HF_TOKEN` environment variable directly

### Training loss not decreasing (Condition C)

With the default `small_random` init, gradients should flow from step 1. If loss is flat after 500 steps:
- Check gradient norm logs in wandb (`grad_norm/inserted_mean`, `grad_norm/inserted_max`)
- If gradient norms are very small, increase learning rate to 5e-5
- If gradient norms are normal but loss is flat, the reward signal may be too sparse — check what fraction of rollouts get non-zero reward

### Slow evaluation

- Use `--test-run` for quick checks (50 examples)
- Full GSM8K test set (1,319 examples) takes ~1 hour with greedy decoding

### TRL version compatibility

This code was written against **TRL v1.1.0** (April 2026). Key API points:
- `GRPOTrainer` accepts `reward_funcs` (plural), `processing_class` (not `tokenizer`), and `peft_config` as a direct parameter
- `GRPOConfig` uses `num_generations` for rollout count, `beta` for KL coefficient
- Reward functions receive completions in chat format: `list[list[dict]]` where each completion is `[{"role": "assistant", "content": "..."}]`
- Ground truth values are passed to reward functions via keyword arguments matching dataset column names

If you're on a different TRL version, check the [TRL changelog](https://github.com/huggingface/trl/releases) for breaking changes.

### KL penalty and memory

Setting `GRPOConfig.kl_coef` (mapped to `beta`) to `0.0` skips loading the reference model entirely, saving significant GPU memory. Useful for initial debugging on smaller GPUs. The default `0.01` does load a reference model.
