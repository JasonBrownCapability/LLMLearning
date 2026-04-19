#!/bin/bash
# Final re-runs for missing per-seed data
# Run from the project root: bash run_final.sh
#
# Estimated total time: ~8 hours

set -e

OUTPUT_DIR="/workspace/results_final"
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
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
python -m experiment.train --condition d --output-dir $OUTPUT_DIR --seeds $SEEDS --insertion-positions 6,12,20,26

echo ""
echo "============================================"
echo "All runs complete!"
echo "Results saved to $OUTPUT_DIR"
echo "============================================"
