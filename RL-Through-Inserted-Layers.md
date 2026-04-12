# Reinforcement Learning Through Inserted Transformer Layers: A Path to Post-Training Reasoning Upgrades

## 1. Introduction

Large Language Models acquire their capabilities during pre-training, a process that is extraordinarily expensive in compute, data, and time. Once training is complete, upgrading a model's *reasoning* abilities — its capacity for multi-step inference, planning, self-correction, and algorithmic thinking — remains an open challenge.

Factual knowledge can be injected at inference time via Retrieval-Augmented Generation (RAG). Behavioural alignment can be achieved through fine-tuning or reinforcement learning. But genuine reasoning improvements — teaching a model to think in ways it couldn't before — require changes to how the model *computes*, not just what it knows.

This document explores a specific approach: **inserting new transformer layers into a frozen pre-trained model, initialised as no-ops, and training only those layers via Reinforcement Learning (RL)**. Each component is individually supported by existing research. Several related combinations have been explored — notably PNN-LLaMA (architectural expansion + LoRA + EWC) and CLS-inspired dual-rate learning systems in vision — but the specific combination of RL-trained inserted layers as a consolidation target within a CLS-inspired framework for LLM reasoning appears to be unexplored. This document synthesises these threads and proposes a concrete architecture.

---

## 2. Background: Post-Training Learning Methods

### 2.1 Full Fine-Tuning
Updates all model weights on new data. Effective but expensive, and risks **catastrophic forgetting** — the model loses previously-learned capabilities as it adapts to new data.

### 2.2 Parameter-Efficient Fine-Tuning (PEFT)
Updates only a small subset of parameters:
- **LoRA / QLoRA** (Hu et al., 2021): Adds low-rank decomposition matrices (A × B) to attention/FFN weight matrices, initialised so their product is zero (a true no-op). Only these are trained. Currently the dominant approach.
- **Adapter Layers** (Houlsby et al., 2019): Small bottleneck modules inserted between transformer sub-layers, initialised near-zero.
- **Prefix/Prompt Tuning**: Learns soft prompt embeddings prepended to input.

### 2.3 Reinforcement Learning from Human/AI Feedback
- **RLHF**: Trains a reward model from human preferences, optimises the LLM via PPO.
- **DPO** (Rafailov et al., 2023): Skips the reward model; optimises preferences directly.
- **GRPO / Rule-based RL**: Rewards based on verifiable outcomes (math correctness, code execution). Used in DeepSeek-R1 and similar reasoning models.
- **REINFORCE++** (2025): Streamlined policy gradient for LLMs, reduced RLHF training time significantly vs PPO.

### 2.4 Retrieval-Augmented Generation (RAG)
Grounds generation in external, updateable knowledge stores. Effective for factual knowledge but does not alter the model's reasoning process.

### 2.5 In-Context Learning
Prompt-based adaptation (few-shot, chain-of-thought). No weight changes; the model adapts behaviour dynamically. Chain-of-thought effectively adds computational depth at inference time via sequential token generation.

### 2.6 Continual / Lifelong Learning
Training on a sequence of tasks without forgetting prior ones. Techniques include elastic weight consolidation (EWC), experience replay, and parameter isolation. Catastrophic forgetting remains the core challenge.

### 2.7 Knowledge Editing
Surgically modifies specific facts in model weights (ROME, MEMIT). Good for individual facts; does not scale to reasoning changes.

### 2.8 Complementary Learning Systems in Deep Learning

Complementary Learning Systems (CLS) theory from neuroscience (McClelland et al., 1995) has already been applied to deep learning for continual learning, though primarily in computer vision rather than LLMs:

- **CLS-ER** (Arani et al., ICLR 2022): Directly implements CLS with dual semantic memories (short-term and long-term) interacting with episodic memory replay. Demonstrated effective continual learning in vision tasks with reduced catastrophic forgetting.
- **DualNet** (Pham et al., NeurIPS 2021): Implements fast (supervised) and slow (self-supervised) learning systems within a single network for continual learning. Tested on vision benchmarks.
- **Generative Replay** (Shin et al., 2017; Sun et al., EACL 2023): Inspired by hippocampal replay, uses the model to generate pseudo-data from past tasks to interleave during new learning. Sun et al. applied this specifically to language models.
- **H.O.P.E. / Nested Learning**: Multi-rate optimisation within a single transformer, where different FFN blocks update at different frequencies — fast-adapting layers for recent information, slow-adapting layers for stable knowledge.

These works establish that CLS-inspired dual-rate architectures are effective for continual learning. The approach proposed in this document (Section 7) builds on this foundation, applying the fast/slow distinction specifically to LLM reasoning using LoRA and inserted layers as the two systems.

---

## 3. The Core Idea: RL Through Inserted Layers

### 3.1 Architecture

Take a pre-trained transformer with N layers. Insert K new transformer layers at selected positions throughout the model. The architecture becomes:

```
[Frozen Layer 1] → [Frozen Layer 2] → ... → [NEW Layer A] → [Frozen Layer i+1] → ... → [NEW Layer B] → ... → [Frozen Layer N]
```

### 3.2 No-Op Initialisation

Each inserted layer must initially produce zero net effect on the model's output. In a standard transformer layer, the residual connection means:

```
output = input + LayerOutput(input)
```

If `LayerOutput(input) = 0` for all inputs, the layer is a no-op. This can be achieved by **initialising the output projection matrices of both the self-attention and FFN sub-layers to zero**. The residual stream passes through unchanged, and the model behaves identically to its pre-insertion state.

### 3.3 Selective Training

- **Freeze** all original model weights entirely.
- **Train only** the K inserted layers.
- The base model's knowledge and existing capabilities are preserved by construction.

### 3.4 Reinforcement Learning as the Training Signal

Rather than supervised fine-tuning on reasoning traces (which teaches imitation), use RL with verifiable rewards:

- **Mathematical reasoning**: reward = correctness of final answer
- **Code generation**: reward = code passes test cases
- **Logical reasoning**: reward = valid conclusion from premises
- **General reasoning**: reward model trained on human preference for reasoning quality

RL allows the model to discover its own reasoning strategies rather than copying human-generated examples. This is the approach that produced the strongest reasoning gains in DeepSeek-R1 and OpenAI's o1-class models.

---

## 4. Supporting Evidence

### 4.1 RL Through Adapters on Frozen Bases — Proven Viable

The feasibility of running RL through a parameter subset while the base model is frozen is well-established:

- **VeRL** (production RL framework) ships with PPO + LoRA support on frozen bases.
- **S-GRPO and T-SPMO** using LoRA adapters (frozen base) increased SVAMP test accuracy from 46% to over 70% on Qwen2-1.5B, demonstrating meaningful reasoning gains through adapter-only RL.
- **REINFORCE++** (2025) treats the frozen reference model as standard practice, achieving significant efficiency gains over PPO while maintaining performance.

This validates the core mechanism: RL gradients can flow through a small set of trainable parameters and produce meaningful behavioural changes, even when the vast majority of the model is frozen.

The closest existing work combining multiple elements of this approach is **PNN-LLaMA**, which uses Progressive Neural Networks (architectural expansion) combined with LoRA and Elastic Weight Consolidation (EWC) for LLM continual learning. PNN-LLaMA demonstrates that the general strategy — architectural expansion + parameter-efficient adaptation + consolidation in an LLM — is viable. The approach proposed here differs in two respects: it uses RL rather than EWC as the consolidation mechanism (targeting reasoning rather than knowledge retention), and uses inserted transformer layers rather than PNN columns as the expansion architecture (adding depth within the forward pass rather than parallel capacity).

### 4.2 Reasoning Is Depth-Dependent — Theoretical Proof

The strongest argument for why inserted layers would outperform LoRA for reasoning comes from computational complexity theory:

- **Feng et al. (2024), "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems"**: Proves that constant-depth transformers are fundamentally limited to problems in AC⁰ or TC⁰ complexity classes. However, with T chain-of-thought steps, transformers can solve problems requiring polynomial time. The key insight: *"without CoT, the number of serial computations is bounded by depth (a fixed constant), whereas with T intermediate steps, serial computations boost to T."*

- **Growing Transformers (2025), "Modular Composition and Layer-wise Expansion on a Frozen Substrate"** (arxiv 2507.07129): Demonstrates progressive layer-wise growth with frozen substrates. Key finding: SQuAD performance jumped from 3.75% at 3 layers to 5.55% at 6 layers. Conclusion: *complex reasoning abilities are an emergent property of model depth*.

- **Sparse Growing Transformer (2025)** (arxiv 2603.23998): Introduces entropy-guided attention looping — selectively allocating additional computational depth to high-entropy attention heads (those distributing attention globally rather than locally). Rather than inserting layers uniformly, it targets positions where the model is most uncertain. This reduced training overhead from 16-20% to just 1-3% while achieving superior reasoning performance across model scales (275M to 1.2B parameters).

**Implication for this approach**: LoRA modifies what existing layers do but **cannot add computational depth**. If a reasoning task requires more sequential processing steps than the model has layers, LoRA cannot solve it. Inserted layers can.

### 4.3 Layer Insertion Mechanics — Established Techniques

- **Net2Net** (Chen, Goodfellow, & Shlens, 2015): Introduced identity-initialised layer insertion as a way to grow networks. Proved that the deeper network starts with identical behaviour to the original.
- **Depth Up-Scaling** (Kim et al., 2024, Solar 10.7B): Took a trained 32-layer model, duplicated selected layers to create a 48-layer model, and continued pre-training. The model performed competitively, validating that trained transformers tolerate post-hoc depth increases.
- **Sparse Upcycling** (Komatsuzaki et al., 2022): Converts dense models to Mixture-of-Experts by inserting expert copies at FFN layers, initialised to preserve original behaviour. Demonstrates that transformers robustly integrate new capacity.

### 4.4 Reasoning via RL — The Frontier

- **DeepSeek-R1-Zero**: Trained via large-scale RL without supervised fine-tuning. Demonstrated that genuine reasoning behaviours (backtracking, self-correction, exploration) can emerge from pure RL. Architecture uses Mixture-of-Experts (671B parameters, 37B activated per token).
- **OpenAI o1/o3**: Details remain opaque, but these models demonstrate substantially improved reasoning, likely through extensive RL and increased test-time compute.
- Both approaches applied RL to the **full model**. Neither explored restricting RL to newly-inserted capacity.

---

## 5. Why This Combination

The theoretical argument chains together:

1. **Reasoning requires serial computational depth** — proved by complexity theory (Feng et al., 2024). A fixed-depth transformer has a hard ceiling on the reasoning complexity it can handle in a single forward pass.

2. **LoRA cannot add depth** — it modifies existing layer computations but adds no new sequential processing steps. For tasks within the model's existing computational budget, LoRA is sufficient. For tasks requiring deeper reasoning chains, it is structurally limited.

3. **Inserted layers add genuine computational depth** — each new layer is a new sequential processing step in the forward pass. This directly expands the class of problems the model can solve.

4. **RL is the strongest method for training reasoning** — supervised fine-tuning teaches imitation of reasoning traces. RL rewards outcomes, allowing the model to discover novel reasoning strategies. The strongest reasoning models (DeepSeek-R1, o1) all use RL.

5. **Freezing the base preserves existing capabilities** — catastrophic forgetting is avoided by construction. The base model's knowledge, language abilities, and existing reasoning remain intact.

6. **Entropy-guided placement optimises the investment** — rather than inserting layers uniformly, targeting positions where the model's attention is most diffuse (uncertain) focuses new capacity where it is most needed.

Therefore: **RL through newly-inserted layers on a frozen base is theoretically the most targeted method to upgrade reasoning capabilities post-training**.

**Positioning relative to prior work**: None of the six points above is individually novel — each builds on established research. The contribution is their specific combination into a single architecture optimised for reasoning. The closest prior art, PNN-LLaMA, combines architectural expansion with LoRA and consolidation, but uses EWC (which preserves knowledge) rather than RL (which discovers new reasoning strategies), and uses parallel columns rather than serial depth (which doesn't expand the model's computational complexity class). The argument here is that for *reasoning specifically*, the choice of RL + serial depth is important — and that specific combination has not been tested.

---

## 6. Open Questions and Challenges

### 6.1 Layer Placement Strategy
Where should new layers be inserted? Options include:
- **Uniform spacing** — simple but likely suboptimal
- **Entropy-guided** — insert where attention entropy is highest (Sparse Growing Transformer approach)
- **Task-dependent** — different reasoning tasks may benefit from depth at different positions
- **Learned placement** — treat insertion position as a hyperparameter to optimise

### 6.2 Training Dynamics
- **Gradient flow through zero-initialised layers**: Gradients to new layers will initially be very small because the output projections are zero. Learning may start very slowly. Mitigation: learning rate warmup, or initialising from small random values rather than exact zero.
- **Interaction between new and frozen layers**: Once new layers start producing non-zero output, they alter the input distribution for all subsequent frozen layers. The residual stream mitigates this (the contribution is additive), but large deviations could still disrupt downstream computation.
- **RL instability**: RL training is inherently less stable than supervised training. Combining RL's instability with the gradient challenges of zero-initialised layers may require careful hyperparameter tuning.

### 6.3 How Many Layers?
- Too few: insufficient new computational depth
- Too many: large parameter cost, slow training, risk of disrupting the residual stream
- The optimal number likely depends on the gap between the model's current reasoning capability and the target task complexity

### 6.4 Evaluation
How to measure "improved reasoning" rigorously:
- Standard benchmarks: GSM8K, MATH, ARC, HumanEval, GPQA
- Comparison baselines: same model + LoRA + RL, same model + full RL fine-tuning
- Ablations: inserted layers trained with supervised learning vs RL, different insertion positions, different numbers of layers

### 6.5 Computational Cost
- Inserted full transformer layers have many more parameters than LoRA adapters
- Forward/backward pass cost increases linearly with number of inserted layers
- Is the reasoning improvement worth the cost compared to simpler methods like chain-of-thought prompting (which adds "depth" at inference time via sequential token generation)?

### 6.6 Relationship to Chain-of-Thought
Chain-of-thought (CoT) also adds serial computation, but at inference time through sequential token generation rather than architectural depth. Key comparison:
- **CoT**: unbounded additional computation, but uses existing circuits repeatedly. Increases inference cost per query.
- **Inserted layers**: bounded additional computation, but creates genuinely new circuits. One-time training cost, minimal per-query overhead.
- These approaches are complementary, not competing. Inserted layers could improve the quality of each reasoning step within a chain-of-thought.

---

## 7. A Two-Stage Learning Architecture: LoRA as Short-Term Memory, Inserted Layers as Long-Term Consolidation

### 7.1 The Neuroscience Parallel: Complementary Learning Systems

Humans do not learn by writing directly into long-term memory. New information enters **short-term memory** first — fast, flexible, and temporary. Through practice and repetition (and notably, sleep), it is gradually **consolidated** into long-term memory and habitual behaviour.

This mirrors **Complementary Learning Systems (CLS) theory** (McClelland, McNaughton, & O'Reilly, 1995; updated by Kumaran, Hassabis, & McClelland, 2016):

- **Hippocampus** — fast, flexible learning. Quickly encodes new experiences with minimal interference. Temporary storage.
- **Neocortex** — slow, structured learning. Through repeated replay (especially during sleep), hippocampal memories are gradually consolidated into stable, long-term cortical representations.

The two systems have different learning rates and architectures for a reason: fast learning directly in the neocortex would cause catastrophic interference with existing knowledge. The hippocampus acts as a buffer that protects long-term knowledge during the consolidation process.

As noted in Section 2.8, CLS has already been applied to deep learning: CLS-ER (Arani et al., 2022) and DualNet (Pham et al., 2021) implement dual-rate systems for continual learning in vision, and generative replay (Sun et al., 2023) brings hippocampal-inspired replay to language models. The **H.O.P.E. / nested learning** architecture implements multi-rate optimisation directly within transformer FFN blocks. What follows extends this line of work to LLM reasoning with a specific architectural proposal: LoRA as the fast system and RL-trained inserted layers as the slow system.

### 7.2 Mapping to LLM Components

| Human System | LLM Equivalent | Properties |
|---|---|---|
| Short-term / hippocampal memory | **LoRA adapters** | Fast to train, lightweight, temporary, easily attached/detached, task-specific |
| Long-term / neocortical memory | **Inserted transformer layers** | Slow to train, permanent, adds computational depth, generalises across contexts |
| Practice / sleep replay | **RL consolidation** | Repeated rollouts from the LoRA-augmented model used to train the inserted layers |
| Forgetting short-term after consolidation | **LoRA removal** | Once the inserted layers capture the skill, the LoRA is discarded |

### 7.3 The Learning Cycle

The two stages feed into each other in a repeating cycle:

**Stage 1 — ACQUIRE (fast, LoRA)**
Train a LoRA adapter on a new reasoning task using supervised examples or short RL runs. The LoRA attaches to the current model (base + any previously-consolidated inserted layers). This is fast and cheap — the equivalent of a human quickly grasping a new concept.

**Stage 2 — GENERATE EXPERIENCE (replay)**
Use the LoRA-augmented model to generate thousands of reasoning rollouts on the target task. Score them via verifiable rewards (mathematical correctness, code execution, logical validity). This is the equivalent of hippocampal replay — curated re-experiencing of successful reasoning.

**Stage 3 — CONSOLIDATE (slow, inserted layers + RL)**
Train the inserted layers via RL (GRPO/PPO) on the scored rollouts. The inserted layers learn to replicate and generalise what the LoRA enabled. This is the slow, deep learning phase — building new computational circuits for the reasoning pattern.

**Stage 4 — DISCARD THE LoRA**
Test whether the model (base + inserted layers, without the LoRA) can now perform the task. If performance holds, consolidation succeeded — the skill is now a permanent part of the model's reasoning architecture. If partial, repeat from Stage 2 with additional rollouts.

**Stage 5 — REPEAT FOR NEXT SKILL**
Attach a new LoRA for the next reasoning capability. The inserted layers now carry all previously-consolidated skills as foundation. Each cycle builds on the last.

### 7.4 Why This Is Better Than Either Mechanism Alone

**LoRA alone** learns fast but shallow. Each LoRA is independent — skills don't compose or build on each other. Multiple LoRAs can conflict. And LoRA cannot exceed the model's existing computational depth, limiting the complexity of reasoning it can capture.

**Inserted layers + RL alone** learns deep but the optimisation problem is hard. Starting RL from zero-initialised layers with sparse rewards means tiny gradients, vast exploration spaces, and slow convergence.

**Combined**, the LoRA acts as a **scaffold**. It quickly gets the model into the right behavioural neighbourhood, generating high-quality rollouts that provide a dense training signal for the inserted layers. This is dramatically easier than learning from scratch — the RL reward landscape is smoother because the LoRA-augmented model is already producing partially-correct reasoning.

This directly parallels how hippocampal replay provides structured training signal to the neocortex — consolidation operates on curated replays of successful experiences, not raw unfiltered input.

**Relation to existing LoRA consolidation work**: The idea of LoRA as temporary learning that gets consolidated is not entirely new. "Merge before Forget" (Qiao & Mahdavi, 2025) trains task-specific LoRAs and merges them into a unified LoRA. CONEC-LoRA (Paeedeh et al., 2025) uses a dual LoRA architecture with task-shared (slow) and task-specific (fast) components. Both treat LoRA as the fast system in a consolidation pipeline. The difference here is the consolidation *target*: rather than merging into another LoRA or into base weights (which doesn't add depth), consolidation targets newly-inserted layers that expand the model's computational capacity. And the consolidation *mechanism* is RL rather than weight averaging — enabling the discovery of novel reasoning strategies rather than just preserving learned ones.

### 7.5 The "Sleep" Phase: Offline Consolidation

In humans, consolidation happens primarily during sleep — offline, with no new input. The LLM equivalent maps naturally to a deployment cycle:

1. **Active phase**: LoRA is attached, model serves queries and accumulates reasoning rollouts
2. **Consolidation phase**: Model goes "offline," inserted layers are trained via RL on accumulated rollouts. Critically, rollouts from **previously-consolidated skills** should be interleaved during training to prevent interference (mirroring how the brain replays old memories alongside new ones during sleep).
3. **Wake phase**: LoRA is removed, model returns to service with upgraded base reasoning capabilities

### 7.6 Cumulative Learning Over Multiple Cycles

After many learning cycles, the model accumulates:

- **Base model** (frozen): Original pre-trained knowledge and capabilities
- **Inserted layers** (slowly evolving): Accumulated reasoning skills from all consolidated cycles — logical deduction, mathematical reasoning, planning, self-correction — each building on previous ones
- **Current LoRA** (temporary): Whatever new skill is currently being acquired

The inserted layers become a living record of learned reasoning capabilities. Skills learned earlier provide foundation for later skills — just as in human learning, where mastering arithmetic enables algebra which enables calculus. The order of consolidation matters.

### 7.7 Open Questions Specific to Two-Stage Learning

- **Curriculum design**: What order should reasoning skills be consolidated in? Earlier consolidations shape the optimisation landscape for later ones. A poor curriculum could make later skills harder to learn.
- **Interference during consolidation**: When training inserted layers on skill B, does skill A degrade? CLS theory suggests interleaved replay of old skills during consolidation prevents this — the same mechanism the brain uses.
- **When to consolidate**: How to determine that a LoRA has captured enough useful signal to warrant the expense of consolidation? Consolidating too early wastes RL compute; too late accumulates stale adapters.
- **Consolidation completeness**: The LoRA-removal test provides a clean measure — if performance holds without the LoRA, the skill is consolidated. But partial consolidation may be acceptable for some skills.
- **Interaction between LoRA and inserted layers during acquisition**: During Stage 1, the LoRA is trained on a model that already has inserted layers from prior cycles. The LoRA may learn to rely on capabilities in those layers, creating a productive interaction where new skills naturally build on consolidated ones.

---

## 8. Assessment: Is This Worth Exploring?

### 8.1 The Convergence Argument

Multiple independent research groups are converging on the same neighbourhood from different directions:

- CLS-ER and DualNet proved fast/slow dual-rate learning works for continual learning
- Merge-before-Forget and CONEC-LoRA proved LoRA-as-temporary-learning with consolidation works
- PNN-LLaMA proved architectural expansion + LoRA + consolidation works in LLMs
- DeepSeek-R1 proved RL produces genuine reasoning capabilities
- Growing Transformers proved depth is what drives reasoning emergence
- Feng et al. proved mathematically that reasoning complexity is bounded by serial depth

When this many threads point toward the same intersection and nobody has connected the final dots, that is usually a sign the idea is *ready* to work, not that it has been overlooked for good reason.

### 8.2 What's Specifically Untested

The gap is narrow but it matters. Every existing consolidation approach uses either:

- **Weight merging** (Merge-before-Forget) — preserves what was learned, doesn't discover anything new
- **EWC** (PNN-LLaMA) — prevents forgetting, doesn't optimise for reasoning outcomes
- **Supervised distillation** — imitates reasoning traces, doesn't explore novel strategies

None uses **RL**, which is the only training signal that has demonstrably produced *new* reasoning behaviours (backtracking, self-correction, multi-step planning) as seen in DeepSeek-R1 and o1.

And none consolidates into **serial depth**, which is the only architectural change that provably expands the complexity class of problems the model can solve.

The untested piece is not a minor variation — it is the piece that theory predicts is specifically necessary for reasoning upgrades.

### 8.3 Risks

**What could go wrong:**

- The RL signal through inserted layers might be too sparse to learn from, even with LoRA scaffolding — the optimisation could simply be too hard
- The gains might not justify the cost over simply using longer chain-of-thought (which adds serial depth at inference time without any training)
- Interference during consolidation might be worse than expected

### 8.4 Why It's Worth Trying

- The experiment is cheap to run at small scale (7B model, 2-4 inserted layers, single GPU)
- Every component has been validated individually — the risk is in the integration, not the fundamentals
- If it works, it solves a real problem: how to permanently upgrade reasoning without full retraining
- A negative result would also be informative — it would tell us something about where the bottleneck in reasoning actually lies (is it depth? is it training signal? is it the interaction between frozen and trainable parameters?)

### 8.5 The Key Experiment

The biggest open question is not whether the architecture works — it is whether **RL through inserted layers produces better reasoning than RL through LoRA alone**. That is the critical comparison. If the answer is yes, it validates the depth argument and the entire approach. If not, it suggests that the existing depth in modern models is already sufficient and the bottleneck lies elsewhere.

---

## 9. References

1. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
2. Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML 2019. arXiv:1902.00751
3. Chen, T., Goodfellow, I., & Shlens, J. (2015). "Net2Net: Accelerating Learning via Knowledge Transfer." arXiv:1511.05641
4. He, J., et al. (2022). "Towards a Unified View of Parameter-Efficient Transfer Learning." ICLR 2022. arXiv:2110.04366
5. Feng, G., et al. (2024). "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems." arXiv:2402.12875
6. "Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate." (2025). arXiv:2507.07129
7. "Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping." (2025). arXiv:2603.23998
8. Kim, D., et al. (2024). "Solar 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling." arXiv:2312.15166
9. Komatsuzaki, A., et al. (2022). "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints." arXiv:2212.05055
10. Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290
11. DeepSeek-AI. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948
12. "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models." (2025). arXiv:2501.03262
13. Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." Anthropic.
14. Olsson, C., et al. (2022). "In-context Learning and Induction Heads." Anthropic.
15. Zhang, J., et al. (2020). "Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks." ECCV 2020. arXiv:1912.13503
16. Gong, L., et al. (2019). "Efficient Training of BERT by Progressively Stacking." ICML 2019. arXiv:1903.11394
17. Pfeiffer, J., et al. (2020). "AdapterHub: A Framework for Adapting Transformers." EMNLP 2020. arXiv:2007.07779
18. Lialin, V., et al. (2023). "Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning." arXiv:2303.15647
19. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex." Psychological Review, 102(3), 419-457.
20. Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). "What Learning Systems Do Intelligent Agents Need? Complementary Learning Systems Theory Updated." Trends in Cognitive Sciences, 20(7), 512-534.
21. Arani, E., Sarfraz, F., & Zonooz, B. (2022). "Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System." ICLR 2022. arXiv:2201.12604
22. Pham, Q., Liu, C., & Hoi, S. (2021). "DualNet: Continual Learning, Fast and Slow." NeurIPS 2021. arXiv:2110.00175
23. Qiao, F. & Mahdavi, M. (2025). "Merge before Forget: A Single LoRA Continual Learning via Continual Merging." arXiv:2512.23017
24. Paeedeh, N., et al. (2025). "CONEC-LoRA: Continual Knowledge Consolidation LoRA." arXiv:2510.16077
25. Sun, F.-K., et al. (2023). "Generative Replay Inspired by Hippocampal Memory Indexing for Continual Language Learning." EACL 2023.
26. Shin, H., et al. (2017). "Continual Learning with Deep Generative Replay." NeurIPS 2017. arXiv:1705.08690

---

*Document compiled April 2026. This document synthesises and extends several active research threads. CLS-inspired dual-rate learning has been implemented for continual learning in vision (CLS-ER, DualNet) and partially for language (generative replay). LoRA-as-temporary-learning with consolidation has been explored (Merge before Forget, CONEC-LoRA). Architectural expansion combined with LoRA and consolidation has been demonstrated in LLMs (PNN-LLaMA). The specific contribution here is the combination of: (a) RL rather than EWC/weight-merging as the consolidation mechanism, targeting reasoning discovery rather than knowledge preservation; and (b) inserted transformer layers rather than parallel columns or merged weights as the consolidation target, adding serial computational depth that expands the model's reasoning capacity. This particular intersection — RL-trained inserted layers as the slow system in a CLS-inspired framework for LLM reasoning — appears to be unexplored as of early 2026, though it builds directly on well-established foundations.*
