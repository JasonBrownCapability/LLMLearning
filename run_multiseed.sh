#!/bin/bash
# Multi-seed runs for key conditions (for arXiv error bars)
# Plus consistent 200-sample evaluations for all conditions
# Run from the project root: bash run_multiseed.sh
#
# Estimated total time: ~8-10 hours (full dataset evaluation)

set -e

OUTPUT_DIR="/workspace/results_multiseed"
WEIGHTS_DIR="/workspace/results"
LORA_PATH="$WEIGHTS_DIR/condition_b_lora_rl"
SEEDS="42,123,456"
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Multi-seed runs starting"
echo "Seeds: $SEEDS"
echo "Eval samples: full dataset"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# ──────────────────────────────────────────────
# Eval-only conditions (no training, single seed)
# ──────────────────────────────────────────────

# Conditions A and B already completed — skip
# To re-run, uncomment the lines below:
# python -m experiment.eval_only --condition a --output-dir $OUTPUT_DIR --benchmarks gsm8k,gsm8k-hard
# python -m experiment.eval_only --condition b --output-dir $OUTPUT_DIR --weights-dir $WEIGHTS_DIR --benchmarks gsm8k,gsm8k-hard

echo ""
echo ">>> Condition C: Inserted layers + RL eval (full dataset)"
echo "============================================"
python -m experiment.eval_only --condition c --output-dir $OUTPUT_DIR --weights-dir $WEIGHTS_DIR --benchmarks gsm8k,gsm8k-hard

# ──────────────────────────────────────────────
# Multi-seed training conditions
# ──────────────────────────────────────────────

echo ""
echo ">>> Condition D: 2 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --insertion-positions 6,12,20,26

echo ""
echo ">>> Condition E: LoRA-merged + 2 inserted + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition e --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS

echo ""
echo ">>> Condition G: LoRA distillation (3 seeds)"
echo "============================================"
python -m experiment.train --condition g --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS

echo ""
echo "============================================"
echo "All runs complete!"
echo "Results saved to $OUTPUT_DIR"
echo "============================================"
