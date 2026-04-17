#!/bin/bash
# Multi-seed runs for key conditions (for arXiv error bars)
# Run from the project root: bash run_multiseed.sh
#
# Estimated total time: ~3-4 hours

set -e

OUTPUT_DIR="/workspace/results"
LORA_PATH="$OUTPUT_DIR/condition_b_lora_rl"
SEEDS="42,123,456"
MAX_EVAL=200

echo "============================================"
echo "Multi-seed runs starting"
echo "Seeds: $SEEDS"
echo "Max eval samples: $MAX_EVAL"
echo "============================================"

echo ""
echo ">>> Condition D: 2 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL --insertion-positions 6,12,20,26

echo ""
echo ">>> Condition G: LoRA distillation (3 seeds)"
echo "============================================"
python -m experiment.train --condition g --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS --max-eval-samples $MAX_EVAL

echo ""
echo "============================================"
echo "All multi-seed runs complete!"
echo "============================================"
