# Experimental Design: RL Through Inserted Layers vs LoRA for Reasoning

## 1. Hypothesis

**Primary hypothesis**: RL training through newly-inserted transformer layers (adding serial computational depth) produces greater reasoning improvements than RL training through LoRA (modifying existing computations), when both operate on a frozen base model with a comparable trainable parameter budget.

**Secondary hypothesis**: A two-stage approach (LoRA acquisition followed by consolidation into inserted layers via RL) converges faster and achieves better results than RL through inserted layers alone.

**Null hypothesis**: LoRA + RL is sufficient for reasoning improvement and the additional complexity of inserted layers provides no meaningful benefit.

---

## 2. Experimental Conditions

Five conditions, each starting from the same frozen base model:

| Condition | Trainable Component | Training Method | Purpose |
|---|---|---|---|
| **A. Baseline** | Nothing | No training | Measures base model reasoning |
| **B. LoRA + RL** | LoRA (r=64, all linear layers) | GRPO | Standard PEFT baseline |
| **C. Inserted Layers + RL** | 2 new transformer layers | GRPO | Core hypothesis test |
| **D. Inserted Layers + SFT** | 2 new transformer layers | Supervised fine-tuning | Isolates RL vs SFT contribution |
| **E. Two-Stage** | LoRA (stage 1) then inserted layers (stage 2) | SFT/short RL then GRPO | Tests the CLS-inspired pipeline |

### 2.1 Parameter Budget Control

A fair comparison requires controlling for trainable parameter count. For Llama 3 8B:

- **Single transformer layer**: ~250M parameters
- **2 inserted layers**: ~500M trainable parameters
- **LoRA r=64 on all linear layers across all 32 layers**: ~500M trainable parameters

This gives a roughly matched parameter budget between conditions B and C, isolating the architectural difference (depth vs width) rather than simply comparing different amounts of capacity.

### 2.2 Why 2 Inserted Layers (Not 4)

- Keeps parameter count close to achievable LoRA configurations for fair comparison
- Fits comfortably on a single A100 80GB
- If 2 layers show a signal, scaling to 4 is a straightforward follow-up
- If 2 layers show nothing, 4 layers are unlikely to change the picture dramatically

---

## 3. Base Model

**Llama 3 8B** (base, not instruct)

Rationale:
- Well-studied, widely available, strong community support
- 32 layers — inserting 2 represents a ~6% depth increase, which is modest enough to avoid disrupting the residual stream but sufficient to test the depth hypothesis
- 8B is small enough for a single GPU with quantisation but large enough to have meaningful reasoning capability to build on
- Using the base model (not instruct) avoids confounding with prior RLHF/DPO alignment

**Quantisation**: Frozen base in 4-bit (NF4 via bitsandbytes). Inserted layers in full fp16 precision.

---

## 4. Layer Insertion Details

### 4.1 Placement

Insert new layers at two positions within the 32-layer model:

- **Position A**: After layer 10 (early-middle — syntactic/semantic boundary)
- **Position B**: After layer 21 (middle-late — semantic/reasoning boundary)

These positions are based on findings from mechanistic interpretability work suggesting that middle layers handle semantic composition and later layers handle task-specific reasoning. An ablation study (Section 7) tests alternative placements.

### 4.2 Architecture of Inserted Layers

Each inserted layer is a standard transformer decoder layer matching the base model's architecture:
- Multi-head self-attention (same head count, dimensions as base)
- SwiGLU FFN (same intermediate dimension as base)
- RMSNorm (pre-norm, matching Llama architecture)
- Rotary position embeddings (shared with base model)

### 4.3 No-Op Initialisation

For each inserted layer:
- Self-attention Q, K, V projections: initialised from Kaiming normal (standard)
- Self-attention output projection (O): **initialised to zero**
- FFN up/gate projections: initialised from Kaiming normal
- FFN down projection: **initialised to zero**
- RMSNorm: initialised to ones (standard)

With O and down projections at zero, the self-attention and FFN sub-layers both produce zero output. The residual connection passes the input through unchanged:
```
output = input + 0 + 0 = input
```

**Verification step**: Before any training, run the modified model on the full GSM8K test set and confirm performance matches the unmodified base model exactly (within floating-point tolerance).

---

## 5. Training Setup

### 5.1 RL Algorithm: GRPO

**Group Relative Policy Optimisation** (GRPO), as used in DeepSeek-R1:

- For each training prompt, generate G=8 completions (rollouts)
- Score each completion with the reward function
- Use the group's mean reward as baseline
- Update policy to increase probability of above-average completions

GRPO avoids the need for a separate critic/value model (unlike PPO), reducing memory requirements.

**Framework**: HuggingFace TRL's `GRPOTrainer`

### 5.2 Reward Function

**Primary reward** (for GSM8K/MATH training): binary correctness
- Extract the final numerical answer from the model's output
- Compare to ground truth
- Reward = 1.0 if correct, 0.0 if incorrect

**Format reward** (small bonus): +0.1 if the model shows intermediate reasoning steps before the answer, to encourage chain-of-thought without prescribing its form.

**KL penalty**: Standard KL divergence penalty against the frozen base model to prevent the policy from drifting too far. Coefficient β = 0.01 (tuned in preliminary runs).

### 5.3 Training Data

**GSM8K training set** (~7.5K problems) as the primary RL training source.

- Problems are grade-school math — achievable for an 8B model but with clear room for improvement
- Answers are numerical and unambiguously verifiable
- Diverse enough to require genuine multi-step reasoning, not pattern matching

### 5.4 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | 1e-5 | With linear warmup over 100 steps |
| Batch size | 4 prompts × 8 rollouts = 32 completions | Per gradient step |
| Max sequence length | 1024 tokens | Sufficient for GSM8K reasoning traces |
| Training steps | 2000 | ~1 epoch over GSM8K training set |
| Gradient checkpointing | Enabled | Required for memory |
| Optimizer | AdamW | β1=0.9, β2=0.999, weight decay 0.01 |
| KL coefficient (β) | 0.01 | |
| Temperature (generation) | 0.7 | For rollout diversity |

### 5.5 Condition-Specific Details

**Condition B (LoRA + RL)**: LoRA r=64 applied to Q, K, V, O, up, gate, down projections across all 32 frozen layers. LoRA alpha=128. Trained with GRPO using same hyperparameters.

**Condition C (Inserted Layers + RL)**: Only the 2 inserted layers have `requires_grad=True`. All 32 base layers fully frozen. Trained with GRPO using same hyperparameters.

**Condition D (Inserted Layers + SFT)**: Same 2 inserted layers, but trained via supervised fine-tuning on GSM8K training set with provided reasoning traces (the "solution" field). Uses standard cross-entropy loss. Same learning rate and steps for comparability.

**Condition E (Two-Stage)**:
- Stage 1: Train a LoRA r=32 via GRPO on GSM8K for 1000 steps
- Stage 2: With the trained LoRA attached, generate 50K rollouts on GSM8K, scored by correctness. Then detach the LoRA, and train the inserted layers via GRPO using these rollouts as the training distribution (on-policy generation from the LoRA-augmented model, off-policy training of the inserted layers). Train for 1000 steps.
- Stage 3: Evaluate with LoRA detached — does performance hold?

---

## 6. Evaluation

### 6.1 Benchmarks

| Benchmark | Type | Size | Purpose |
|---|---|---|---|
| **GSM8K test** | Math word problems | 1,319 | In-distribution reasoning |
| **MATH** | Competition maths | 5,000 | Out-of-distribution difficulty scaling |
| **ARC-Challenge** | Science reasoning (MC) | 1,172 | Transfer to non-mathematical reasoning |
| **LogiQA** | Logical reasoning (MC) | 651 (test) | Transfer to formal logical reasoning |
| **HumanEval** | Code generation | 164 | Transfer to procedural/algorithmic reasoning |

### 6.2 Metrics

For each condition and benchmark:
- **Accuracy** (pass@1): single greedy-decoded answer, scored for correctness
- **Accuracy** (pass@8): best of 8 sampled completions — measures whether the model *can* solve problems it doesn't reliably solve
- **Reasoning trace quality** (qualitative): manual inspection of 50 randomly-selected reasoning traces per condition, looking for:
  - Multi-step decomposition
  - Self-correction / backtracking
  - Novel reasoning strategies (not seen in training data)

### 6.3 Key Comparisons

| Comparison | What It Tests |
|---|---|
| C vs B | **Core hypothesis** — does added depth (inserted layers) beat added width (LoRA) for reasoning? |
| C vs D | Does RL produce better reasoning than SFT when training inserted layers? |
| C vs A | Do inserted layers + RL improve reasoning at all over the base model? |
| E vs C | Does LoRA scaffolding (two-stage) improve convergence or final performance? |
| E stage 3 vs E stage 2 | Does consolidation succeed — does performance hold when the LoRA is removed? |
| B vs A | Sanity check — does LoRA + RL improve reasoning? (Expected yes, based on prior work) |

---

## 7. Ablation Studies

Run after the main experiment, only if the core hypothesis (C > B) is supported:

### 7.1 Layer Placement
Test 2 inserted layers at different positions:
- Both early (after layers 5, 10)
- Both middle (after layers 12, 20)
- Both late (after layers 24, 28)
- One early, one late (after layers 8, 24)

### 7.2 Number of Layers
Compare 1, 2, 3, 4 inserted layers (adjusting LoRA rank in condition B to match parameter count each time).

### 7.3 Initialisation Strategy
- Zero-init (default)
- Small random init (output projections from N(0, 0.01))
- Copy-init (duplicate an existing layer's weights, as in Solar depth up-scaling)

### 7.4 Reward Complexity
- Binary correctness only
- Correctness + step-count bonus (reward more concise reasoning)
- Correctness + process reward model (if available — scores intermediate reasoning steps)

---

## 8. Expected Outcomes and Interpretation

### 8.1 If C > B (inserted layers beat LoRA)
The depth hypothesis is supported. Reasoning improvement benefits from additional serial computation that LoRA cannot provide. This would be a significant finding worth reporting, and justifies exploring the two-stage CLS approach further.

### 8.2 If C ≈ B (comparable performance)
Depth is not a bottleneck for the tested reasoning tasks at this model scale. Possible explanations: (a) 8B models already have sufficient depth for GSM8K-level reasoning, (b) LoRA's ability to rewire existing circuits is as effective as new circuits for this difficulty level. **Follow-up**: test on harder benchmarks (MATH level 5, GPQA) where depth limitations might be more apparent.

### 8.3 If C < B (LoRA beats inserted layers)
The optimisation challenge of training new layers from zero-init outweighs the theoretical depth advantage. Possible explanations: (a) RL signal is too sparse to train fresh layers effectively, (b) zero-init creates a gradient bottleneck that prevents learning within the training budget. **Follow-up**: test condition E (two-stage) which specifically addresses the optimisation difficulty via LoRA scaffolding. If E > C, the architecture has merit but needs the scaffolding. If E ≈ C, the approach may not be viable.

### 8.4 If D ≈ C (SFT matches RL for inserted layers)
RL's advantage over SFT may not extend to inserted-layer training. This would suggest the inserted layers primarily learn to *reproduce* existing reasoning patterns rather than *discover* new ones. This would be a meaningful negative result for the RL component specifically.

### 8.5 If E > C (two-stage beats direct RL)
The CLS-inspired pipeline adds value — LoRA scaffolding makes the optimisation landscape easier for inserted layer training. This validates the biological analogy and suggests a practical training pipeline for production use.

---

## 9. Infrastructure and Timeline

### 9.1 Hardware
- **Minimum**: 1x A100 80GB (or equivalent: H100, 2x A6000 48GB)
- **Comfortable**: 2x A100 80GB (allows larger rollout groups and longer sequences)
- **Cloud estimate**: ~$2/hr for 1x A100 on most cloud providers

### 9.2 Estimated Compute Time

| Step | Time (1x A100) |
|---|---|
| Condition A (baseline eval) | ~1 hour |
| Condition B (LoRA + RL training) | ~8-12 hours |
| Condition C (inserted layers + RL training) | ~10-14 hours |
| Condition D (inserted layers + SFT) | ~4-6 hours |
| Condition E (two-stage) | ~12-16 hours |
| Full evaluation (all conditions, all benchmarks) | ~6-8 hours |
| **Total main experiment** | **~40-55 hours** |
| Ablation studies (if warranted) | ~30-40 hours additional |

**Total cloud cost estimate**: ~$100-200 for the main experiment. ~$60-80 additional for ablations.

### 9.3 Software Stack

- **Model**: `meta-llama/Meta-Llama-3-8B` via HuggingFace
- **Quantisation**: `bitsandbytes` (4-bit NF4)
- **RL training**: `trl` (HuggingFace TRL, `GRPOTrainer`)
- **LoRA**: `peft` (HuggingFace PEFT)
- **Layer insertion**: Custom PyTorch (~50-100 lines)
- **Evaluation**: `lm-evaluation-harness` (EleutherAI) for standardised benchmark evaluation
- **Logging**: `wandb` for training curves and comparison

### 9.4 Implementation Order

1. **Setup and verification** (~1 day)
   - Load Llama 3 8B in 4-bit
   - Implement layer insertion code with zero-init
   - Verify no-op property: modified model matches base model outputs exactly
   - Run baseline evaluation (Condition A)

2. **Condition B: LoRA + RL** (~1 day)
   - Configure GRPOTrainer with LoRA
   - Train on GSM8K
   - Evaluate on all benchmarks
   - This serves as both a baseline and a sanity check that the RL pipeline works

3. **Condition C: Inserted layers + RL** (~1 day)
   - Configure GRPOTrainer with custom trainable parameters
   - Train on GSM8K
   - Evaluate on all benchmarks
   - **This is the core result**

4. **Condition D: Inserted layers + SFT** (~0.5 days)
   - Standard supervised training with same inserted layers
   - Evaluate — isolates RL contribution

5. **Condition E: Two-stage** (~1-2 days)
   - Stage 1: LoRA + RL
   - Rollout generation
   - Stage 2: RL on inserted layers using rollouts
   - Evaluate with and without LoRA

6. **Analysis and ablations** (~2-3 days if warranted)

**Total estimated timeline**: ~7-10 days of active work, assuming hardware is available.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Zero-init gradient vanishing | Monitor gradient norms during training. If too small after 200 steps, switch to small random init (ablation 7.3) |
| RL training instability | Log reward curves closely. If diverging, reduce learning rate or increase KL penalty |
| Base model already too good at GSM8K | Check baseline accuracy first. If >60%, switch to MATH as training source (harder, more headroom) |
| Inserted layers disrupt frozen layers | Monitor base model capability (measure perplexity on a held-out text set) throughout training |
| Memory overflow on single GPU | Reduce rollout group G from 8 to 4, enable gradient accumulation over 2 steps |
| Results are noisy / inconclusive | Run each condition 3 times with different random seeds, report mean and standard deviation |
