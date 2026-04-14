# Cloud GPU Recommendations

## Compute Estimate

With 3 seeds for conditions A/B/C and single runs for D/E:

| Condition | Runs | Hours each | Total |
|-----------|------|-----------|-------|
| A (baseline eval) | 3 | ~1h | ~3h |
| B (LoRA + GRPO) | 3 | ~10h | ~30h |
| C (Inserted + GRPO) | 3 | ~12h | ~36h |
| D (Inserted + SFT) | 1 | ~5h | ~5h |
| E (Two-stage) | 1 | ~14h | ~14h |
| **Total** | | | **~88h** |

## Recommendation: RunPod

**RunPod** is the best fit for this workload:

- **A100 80GB on-demand**: ~$1.64/hr — ~$144 total
- **A100 80GB spot/community**: ~$1.00-1.30/hr — ~$88-114 total
- Persistent volume storage for checkpoints (if a run crashes at step 1800 you don't lose everything)
- Easy to set up with their PyTorch template, `pip install -r requirements.txt`, and go
- You can stop/restart the instance between conditions to avoid paying for idle time

### Why not the alternatives

- **Lambda Cloud**: Cheaper (~$1.10/hr) but A100 80GB instances are frequently sold out. Worth trying first, but don't count on availability.
- **Vast.ai**: Cheapest (<$1/hr) but machines are community-hosted — variable reliability, networking issues, and machines can be reclaimed mid-run on spot. Risky for 12-hour training runs.
- **GCP/AWS/Azure**: $3-5/hr for A100. Works out to $260-440 — 2-3x the cost for the same GPU with no real benefit for this workload.
- **Modal**: Pay-per-second is great for bursty work but 88 continuous hours won't benefit from serverless pricing.

## GPU Choice

**A100 80GB** is the right call.

- **A100 80GB > A100 40GB**: The 80GB gives headroom for GRPO's 8 rollouts per prompt. With 4-bit base model (~5GB) + inserted layers (~1GB fp32 equiv) + 8 rollouts x optimizer states + KL reference model, you'll use 50-65GB. A 40GB card would require reducing rollouts to 4.
- **H100 80GB**: ~$2.50/hr on RunPod. Faster (maybe 30% on this workload due to better bf16 throughput), but the per-hour premium doesn't pay for itself — you'd spend ~$170 for ~65 hours vs ~$144 for ~88 hours on A100.
- **2x A6000 48GB**: Multi-GPU adds complexity with TRL/GRPO and `device_map="auto"` across cards can be finicky. Not worth the debugging risk.

## Practical Tips

- **Run conditions sequentially, not in parallel** — stop the instance between runs if you're stepping away to avoid paying for idle time.
- **Use persistent volumes** — attach a volume for `/results` and model cache (`~/.cache/huggingface`). Downloading Llama 3 8B takes 15-20 min; caching it saves that on every restart.
- **Run A first** as a smoke test (1 hour, confirms setup), then B and C (the core comparison), then decide on D and E based on results.
- **Spot instances are fine for Condition A** (short run, easy to restart) but consider on-demand for the 10-14 hour training runs (B, C) to avoid mid-run preemption.
