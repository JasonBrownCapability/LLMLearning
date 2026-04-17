#!/bin/bash
# Multi-seed runs for key conditions (for arXiv error bars)
# Plus consistent 200-sample evaluations for all conditions
# Run from the project root: bash run_multiseed.sh
#
# Estimated total time: ~4-5 hours

set -e

OUTPUT_DIR="/workspace/results_multiseed"
LORA_PATH="/workspace/results/condition_b_lora_rl"
SEEDS="42,123,456"
MAX_EVAL=200

mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Multi-seed runs starting"
echo "Seeds: $SEEDS"
echo "Max eval samples: $MAX_EVAL"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# ──────────────────────────────────────────────
# Eval-only conditions (no training, single seed)
# ──────────────────────────────────────────────

echo ""
echo ">>> Condition A: Baseline eval (200 samples)"
echo "============================================"
python -m experiment.eval_only --condition a --output-dir $OUTPUT_DIR --max-samples $MAX_EVAL --benchmarks gsm8k,gsm8k-hard

echo ""
echo ">>> Condition B: LoRA eval (200 samples)"
echo "============================================"
python -m experiment.eval_only --condition b --output-dir $OUTPUT_DIR --max-samples $MAX_EVAL --benchmarks gsm8k,gsm8k-hard

echo ""
echo ">>> Condition C: Inserted layers + RL eval (200 samples)"
echo "============================================"
python -m experiment.eval_only --condition c --output-dir $OUTPUT_DIR --max-samples $MAX_EVAL --benchmarks gsm8k,gsm8k-hard

# ──────────────────────────────────────────────
# Multi-seed training conditions
# ──────────────────────────────────────────────

echo ""
echo ">>> Condition D: 2 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL --insertion-positions 6,12,20,26

echo ""
echo ">>> Condition E: LoRA-merged + 2 inserted + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition e --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL

echo ""
echo ">>> Condition G: LoRA distillation (3 seeds)"
echo "============================================"
python -m experiment.train --condition g --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL

echo ""
echo "============================================"
echo "All runs complete!"
echo "Results saved to $OUTPUT_DIR"
echo "============================================"
