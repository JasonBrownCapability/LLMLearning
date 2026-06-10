# RL Through Inserted Transformer Layers — Experiment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20618627.svg)](https://doi.org/10.5281/zenodo.20618627)

## Executive Summary

When you want to teach an LLM new skills (like math reasoning), the standard approach is **LoRA**: attach small adapter weights to every existing layer, tweaking how the model already thinks. This experiment tests an alternative: **inserting brand new layers** into the middle of the model. The existing 32 layers are frozen completely and only the 2 new layers learn, giving the model extra "thinking steps" rather than changing existing ones.

**The core question:** Which learns reasoning better — modifying existing computations (LoRA) or adding new computational depth (inserted layers)?

Both approaches are trained on the same math problems using the same reinforcement learning method (GRPO), with roughly the same number of trainable parameters (~500M) for a fair comparison. Five conditions isolate the key variables:

| Condition | What it does | Why |
|-----------|-------------|-----|
| **A** | No training — evaluate the base model as-is | Establishes baseline performance |
| **B** | LoRA + RL | The standard approach (main comparison) |
| **C** | Inserted layers + RL | The core hypothesis |
| **D** | Inserted layers + supervised learning | Does RL specifically matter, or would any training method work? |
| **E** | LoRA first, then inserted layers | Does strengthening the base model first help the new layers learn? |

The intuition is that inserted layers add *serial depth* (more sequential processing steps), which might be better for step-by-step reasoning than LoRA's *parallel modifications* across existing layers.

See `RL-Through-Inserted-Layers.md` for the full research document and `Experiment-Design.md` for the experimental design rationale.

## Paper & Results

The full write-up is in **[`paper.md`](paper.md)**: *Inserted Layers + SFT Match LoRA + RL on GSM8K at 60× Less Compute, and Can Be Trained Without Re-using Gold Answers via Distillation* (Llama-3.1 8B and Llama-3.2 3B).

**Headline findings:**

- **The going-in hypothesis is *not* supported.** Inserted layers + GRPO (C) reaches 50.3% on GSM8K vs LoRA + GRPO (B) at 59.7% on 8B — a clean negative result.
- **The surprise:** replacing RL with plain SFT on the same inserted layers (D) reaches **59.4% in ~22 minutes** vs B's ~22 hours — within seed noise, at **~60× lower compute**.
- **Distillation (G)** trains inserted layers to match the LoRA teacher's logits using no gold answers in the gradient, recovering ~91% of B's accuracy.
- On the smaller **3B** base, 2 inserted layers + SFT statistically tie B at ~30× less compute; a 4-layer variant exceeds B at 4× the trainable parameters.

Caveats are stated loudly in the paper: the comparison is **compute-matched, not parameter-matched** (inserted recipes carry ~2.6× more trainable params than LoRA r=64), and the key 8B B/C/D-2L runs are single-seed.

**Released alongside the paper:**

- `paper_figures/` — the three figures (PNG + SVG) and the scripts that regenerate them.
- `results*/` — per-example GSM8K / GSM8K-Hard evaluations, summaries, and multi-seed aggregates at the paths the paper cites (Appendix A).
- `results/wandb_logs.tar.gz` — per-step training logs behind Figure 3.

**Regenerate the figures** from the committed data (no GPU, no large artifacts needed):

```bash
python paper_figures/extract_training_curves.py   # wandb logs -> training_curves.json (Figure 3 data)
python paper_figures/build_figures.py             # -> fig1 (compute–accuracy), fig2 (architecture), fig3 (training dynamics)
```

## Prerequisites

- **Python** 3.10+
- **CUDA** 11.8+ with a compatible GPU
- **GPU**: Minimum 1x A100 80GB (or 2x A6000 48GB). For test runs, a 24GB GPU may work.
- **HuggingFace account** with access to Llama 3.1 8B (request access at https://huggingface.co/meta-llama/Llama-3.1-8B)
- **Weights & Biases account** (optional, for logging)

## Local Setup (smoke testing only)

### 1. Install Python 3.11

On Ubuntu 22.04, Python 3.11 isn't in the default repos:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### 2. Create a virtual environment and install dependencies

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Log in to HuggingFace

Create a token at HuggingFace > Settings > Access Tokens with **gated repo read access** (required for Llama 3.1).

```bash
hf auth login
```

### 4. Run the smoke test

```bash
python -m experiment.train --condition a --smoke-test
```

This uses a tiny model (SmolLM-135M) on CPU — no GPU required. See [Local smoke test](#local-smoke-test-no-gpu-required) for details.

## RunPod Setup

Recommended template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` on an **A100 SXM 80GB** with a **Network Volume** (50GB) attached. Enable SSH access and add your public SSH key.

### 1. Connect and set up tmux

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
apt update && apt install tmux -y
tmux new -s train
```

> **Note:** tmux must be reinstalled after each pod restart. To automate this, run:
> `echo 'command -v tmux >/dev/null || apt install tmux -y' >> ~/.bashrc`

### 2. Clone the repo to persistent storage

Clone to `/workspace` so the repo survives pod restarts:

```bash
cd /workspace
git clone https://github.com/JasonBrownCapability/LLMLearning.git
cd LLMLearning
```

> **Note:** This is a public repo, so no authentication is needed to clone.

### 3. Set up persistent storage

```bash
# Cache HuggingFace models on the persistent volume (and persist across restarts)
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc

# Fix CUDA library path for bitsandbytes (and persist across restarts)
echo 'export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc

# Create a results directory on the persistent volume
mkdir -p /workspace/results
```

> **Note:** The `~/.bashrc` lines only need to be added once. On subsequent pod restarts, they are set automatically.

### 4. Install dependencies

Do **not** use a virtual environment on RunPod — the system Python already has PyTorch and CUDA installed. Install the remaining packages directly:

```bash
pip install -r requirements.txt
```

The RunPod template ships with PyTorch 2.4, but TRL 1.1.0 requires PyTorch 2.6+. Upgrade torch and torchvision:

```bash
pip install trl==1.1.0 torch torchvision --upgrade
```

### 5. Log in to HuggingFace

```bash
hf auth login
```

Paste your HuggingFace token (needs gated repo read access for Llama 3.1).

### 6. (Optional) Log in to Weights & Biases

```bash
wandb login
```

Or pass `--no-wandb` to all commands to skip logging.

### 7. Run the test

```bash
python -m experiment.train --condition a --test-run --no-wandb --output-dir /workspace/results
```

> **Tip:** Always pass `--output-dir /workspace/results` so results are saved to the persistent volume. Without this, results are lost if the pod stops.

## Running the Experiment

All conditions are run via the same script. Run from the project root directory (`LLMLearning/`).

### Local smoke test (no GPU required)

```bash
python -m experiment.train --condition a --smoke-test
```

This uses a tiny model (SmolLM-135M) on CPU to verify imports, config, data pipeline, and training loop wiring without needing a GPU. Run this locally before deploying to a cloud GPU to catch setup issues for free.

You can smoke-test any condition:

```bash
python -m experiment.train --condition c --smoke-test   # tests insert + train + eval
```

### Quick test on GPU (verify everything works)

```bash
python -m experiment.train --condition a --test-run --no-wandb --output-dir /workspace/results
```

This loads the full Llama 3.1 8B model, runs pass@1 evaluation on 50 GSM8K examples, and exits. Requires a GPU. Should complete in a few minutes and confirms your GPU setup is correct.

### Condition A: Baseline

Evaluate the unmodified base model (no training).

```bash
python -m experiment.train --condition a --output-dir /workspace/results
```

### Condition B: LoRA + GRPO

Train LoRA adapters via GRPO on GSM8K. This is the main baseline to beat.

```bash
python -m experiment.train --condition b --output-dir /workspace/results
```

Estimated time: 8-12 hours on 1x A100.

### Condition C: Inserted Layers + GRPO

**This is the core experiment.** Insert 2 new transformer layers (small-random-initialised, near-zero effect) and train them via GRPO.

```bash
python -m experiment.train --condition c --output-dir /workspace/results
```

Estimated time: 10-14 hours on 1x A100.

### Condition D: Inserted Layers + SFT

Control condition: same inserted layers, but trained via supervised fine-tuning instead of RL. This isolates the RL vs SFT contribution.

```bash
python -m experiment.train --condition d --output-dir /workspace/results
```

Estimated time: 4-6 hours on 1x A100.

### Condition E: Two-Stage (LoRA then Inserted Layers)

Two-stage training: LoRA via GRPO first (to strengthen the base), then merge LoRA into base weights and train inserted layers via GRPO on the improved model.

```bash
python -m experiment.train --condition e --output-dir /workspace/results
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
python -m experiment.train --condition c --seeds 42,123,456 --output-dir /workspace/results
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

- Confirm you have accepted the Llama 3 licence at https://huggingface.co/meta-llama/Llama-3.1-8B
- Confirm `hf auth login` succeeded
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

This code requires **TRL v1.1.0**, which in turn requires **PyTorch 2.6+**. The RunPod PyTorch 2.4 template ships with an older torch, so you must upgrade (see [RunPod Setup step 4](#4-install-dependencies)). Key API points:
- `GRPOTrainer` accepts `reward_funcs` (plural), `processing_class` (not `tokenizer`), and `peft_config` as a direct parameter
- `GRPOConfig` uses `num_generations` for rollout count, `beta` for KL coefficient
- Reward functions receive completions in chat format: `list[list[dict]]` where each completion is `[{"role": "assistant", "content": "..."}]`
- Ground truth values are passed to reward functions via keyword arguments matching dataset column names

If you're on a different TRL version, check the [TRL changelog](https://github.com/huggingface/trl/releases) for breaking changes.

### bitsandbytes CUDA library error

If you see `libnvJitLink.so.13: cannot open shared object file`, the CUDA libraries installed by pip aren't on the library path. Fix with:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

This is already included in [RunPod Setup step 3](#3-set-up-persistent-storage) but may need re-running if `.bashrc` wasn't sourced.

### KL penalty and memory

Setting `GRPOConfig.kl_coef` (mapped to `beta`) to `0.0` skips loading the reference model entirely, saving significant GPU memory. Useful for initial debugging on smaller GPUs. The default `0.01` does load a reference model.
