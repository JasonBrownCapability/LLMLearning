#!/bin/bash
# Final re-runs for missing per-seed data
# Run from the project root: bash run_final.sh
#
# Estimated total time: ~12 hours

set -e

OUTPUT_DIR="/workspace/results_final"
LORA_PATH="/workspace/results/condition_b_lora_rl"
SEEDS="42,123,456"

mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Final runs for per-seed results"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

echo ""
echo ">>> Condition D: 2 inserted layers + SFT (3 seeds)"
echo "============================================"
# D 2-layer already complete — skip
# python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS

# Clean caches between runs to prevent disk full
cleanup() { rm -rf /root/.cache/* /tmp/pip-* 2>/dev/null; echo "Cache cleaned"; }

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --insertion-positions 6,12,20,26

echo ""
echo ">>> Condition E: LoRA-merged + 2 inserted + SFT (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition e --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS

echo ""
echo ">>> Condition G: LoRA distillation (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition g --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS

echo ""
echo "============================================"
echo "All runs complete!"
echo "Results saved to $OUTPUT_DIR"
echo "============================================"
