# Experiment Results: RL Through Inserted Transformer Layers

## Summary

This experiment tested whether inserting new transformer layers into a frozen base model and training only those layers can produce reasoning improvements comparable to LoRA, the standard parameter-efficient fine-tuning approach. We evaluated multiple training methods (RL, supervised fine-tuning, and a novel distillation approach) across several conditions.

**Key finding:** 4 inserted layers trained via SFT for 25 minutes achieve 58.5% on GSM8K, nearly matching LoRA + RL's 60% achieved over 22 hours of training. A novel distillation method (Condition G) successfully transfers LoRA knowledge into 2 inserted layers, achieving 55% without requiring labelled training data.

## Experimental Setup

- **Base model:** Llama 3.1 8B (base, not instruct), 4-bit quantized (NF4)
- **Hardware:** 1x NVIDIA A100 SXM 80GB (RunPod)
- **Training data:** GSM8K training set (7,473 grade-school math word problems)
- **Evaluation data:** GSM8K test set (1,319 problems), GSM8K-Hard (1,319 problems with larger numbers), MATH (5,000 competition-level problems)
- **Inserted layer architecture:** Full LlamaDecoderLayer (multi-head attention + SwiGLU FFN + RMSNorm), zero-init output projections for no-op start

## Results

### GSM8K (In-Distribution)

| Condition | Description | Trainable Params | GSM8K pass@1 | Training Time |
|-----------|-------------|-----------------|--------------|---------------|
| A | Baseline (no training) | 0 | 18.0% | - |
| C | 2 inserted layers + RL (GRPO) | 436M (8.8%) | 50.4% | ~20 hours |
| D | 2 inserted layers + SFT | 436M (8.8%) | 53.9% | 22 minutes |
| G | 2 inserted layers + LoRA distillation | 436M (8.8%) | 55.0% | ~20 minutes |
| E | LoRA-merged base + 2 inserted layers + SFT | 436M (8.8%) | 56.0% | 15 minutes* |
| D (4 layers) | 4 inserted layers + SFT | 872M (16.1%) | 58.5% | 25 minutes |
| B | LoRA (r=64, all layers) + RL (GRPO) | ~500M | 59.8% | 22 hours |

*Plus condition B's training time for the reused LoRA weights.

### GSM8K-Hard (Out-of-Distribution)

| Condition | Description | GSM8K-Hard pass@1 |
|-----------|-------------|-------------------|
| A | Baseline (no training) | ~4% |
| C | 2 inserted layers + RL (GRPO) | ~16% |
| G | 2 inserted layers + LoRA distillation | 18% |
| D (4 layers) | 4 inserted layers + SFT | 19% |
| D | 2 inserted layers + SFT | ~20% |
| E | LoRA-merged base + 2 inserted layers + SFT | 20% |
| B | LoRA (r=64, all layers) + RL (GRPO) | ~21% |

### MATH (Competition-Level)

All conditions scored below 1% on the MATH benchmark. Competition-level mathematics is beyond the capability of this model regardless of training method.

### Condition B Additional Metrics

Condition B (LoRA + RL) was also evaluated with pass@8 sampling:
- **GSM8K pass@1:** 59.8%
- **GSM8K pass@8:** 91.1%

The pass@8 result indicates the model has learned reasoning strategies that work most of the time but aren't always applied consistently.

## Key Findings

### 1. Inserted layers work for reasoning improvement

Starting from an 18% baseline, 2 inserted layers trained via SFT achieve 54% accuracy on GSM8K. This demonstrates that adding serial computational depth to a frozen model — without modifying any existing weights — produces meaningful reasoning improvements.

### 2. SFT outperforms RL for inserted layers

Supervised fine-tuning consistently outperformed reinforcement learning (GRPO) for training inserted layers:
- **SFT (Condition D):** 54% in 22 minutes
- **RL (Condition C):** 50% in ~20 hours

This is a ~50x speedup with better accuracy. The likely explanation is that RL's sparse reward signal (binary correct/incorrect) is insufficient for layers that start from zero and must learn through frozen base layers. SFT provides a direct, dense training signal from worked solutions.

### 3. More layers help

Doubling from 2 to 4 inserted layers improved accuracy from 54% to 58.5%, nearly closing the gap with LoRA (60%). The positions used were [6, 12, 20, 26], distributing layers across early, mid-early, mid-late, and late stages of the model.

### 4. LoRA distillation works (Condition G)

A novel approach — training inserted layers to match the LoRA model's output logits via KL divergence — achieved 55% accuracy. This demonstrates that LoRA's knowledge, distributed across 32 adapted layers, can be compressed into 2 inserted layers. Notably, this method requires no labelled training data; the training signal comes entirely from the teacher model's output distribution.

### 5. LoRA scaffolding provides marginal benefit for SFT

Condition E (LoRA-merged base + inserted layers + SFT) scored 56% vs Condition D's 54%. The improvement is modest because SFT trains on provided solutions, not model-generated ones. The base model's quality matters less when the training signal comes from the dataset rather than from the model's own completions. LoRA scaffolding would likely help more with RL training (where richer rollouts provide better learning signal).

### 6. All methods generalise similarly to harder problems

On GSM8K-Hard (same problem structure, larger numbers), all trained models scored 16-21%, compared to 4% baseline. The gap between methods narrows substantially on out-of-distribution problems, suggesting all approaches learn similar reasoning patterns that are partially — but not fully — robust to distribution shift.

## Training Observations

### Condition B (LoRA + RL)
- Training reward climbed steadily from 0.20 to 0.75 over 2000 steps
- KL divergence remained stable at 0.10-0.13 throughout training
- Gradient norms stable at 0.5-1.0
- Step time: 35-50s, decreasing over training as completions became shorter

### Condition C (Inserted Layers + RL)
- Required zero-init (small_random init caused model degeneration — KL spiked to 8.3)
- Required 10x lower learning rate (1e-6 vs 1e-5) to prevent instability
- Gradient norms were high (10-40) compared to LoRA (0.5-1.0)
- Training reward plateaued around 0.45-0.55 (vs LoRA's 0.75)
- KL divergence gradually increased to 0.15-0.20, higher than LoRA's stable 0.10-0.13

### Condition D (Inserted Layers + SFT)
- Loss dropped from 1.76 to 0.82 over 2000 steps
- Token accuracy reached 78-79%
- Training completed in 22 minutes (1.14 iterations/second)
- Stable throughout, no degeneration

### Condition G (LoRA Distillation)
- KL divergence loss dropped from 101 to ~1.5 over 2000 steps
- Student model progressively matched teacher's output distribution
- Training completed in ~20 minutes

## Infrastructure Notes

- **GPU:** A100 SXM 80GB. Training used 7-16GB VRAM depending on condition.
- **PyTorch:** Upgraded from 2.4.0 (RunPod template) to 2.11.0 for TRL 1.1.0 compatibility.
- **Key issues encountered:**
  - 4-bit quantized layers cannot be deepcopied; fresh `LlamaDecoderLayer` must be constructed from config
  - Layer indices must be reindexed after insertion for correct KV-cache slot assignment during generation
  - `_hf_peft_config_loaded` flag needed to bypass trainer's quantization validation for non-PEFT trainable components
  - LoRA merging into 4-bit weights loses most learned knowledge due to re-quantization; merge must be done at full precision (bfloat16)
  - Checkpoint saving crashes with the `_hf_peft_config_loaded` hack; auto-saving disabled for inserted layer conditions

## Directions for Further Work

1. **Multi-seed runs:** Run conditions C, D, and the 4-layer variant with seeds 42, 123, 456 for error bars on the key results.

2. **Layer count scaling:** Test 6 and 8 inserted layers to see if the trend continues toward (or beyond) LoRA performance.

3. **Layer placement ablation:** Test different position strategies — both early, both late, concentrated in the middle — to understand where added depth helps most.

4. **Partial freezing (Condition F):** Freeze only layers outside the insertion region, letting neighbouring layers co-adapt with inserted layers.

5. **LoRA + SFT comparison:** Run LoRA with SFT (instead of RL) to determine whether RL benefits LoRA specifically or if SFT is universally better for this task.

6. **Larger models:** Test on Llama 3.1 70B to see if the inserted-layer approach scales.

7. **Combined distillation + SFT:** Train inserted layers with a mixture of KL distillation loss and SFT loss for potentially better results.

## Conclusion

Inserting new transformer layers into a frozen base model is a viable method for improving reasoning capability. While LoRA achieves the highest accuracy (60%), 4 inserted layers trained via SFT reach 58.5% in 25 minutes — a ~50x training speedup with comparable results. The novel distillation approach (Condition G) demonstrates that LoRA's distributed knowledge can be compressed into just 2 inserted layers by matching output distributions, achieving 55% without requiring labelled training data.

These results suggest that adding computational depth — giving the model more "thinking steps" — is a surprisingly effective alternative to modifying existing computations via adapters. The practical implications are significant: inserted layers modify no existing weights, can be swapped in and out, and train in minutes rather than hours.
