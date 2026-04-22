#!/bin/bash
# Resume 3B runs from where run_3b.sh crashed
# A, B, and D 2-layer seed 42 already complete
# Run from the project root: bash run_3b_resume.sh
#
# Estimated remaining time: ~40 hours

set -e

OUTPUT_DIR="/workspace/results_3b"
MODEL="meta-llama/Llama-3.2-3B"
LORA_PATH="$OUTPUT_DIR/condition_b_lora_rl"

mkdir -p $OUTPUT_DIR

cleanup() { rm -rf /root/.cache/* /tmp/pip-* 2>/dev/null; echo "Cache cleaned"; }

echo "============================================"
echo "Resuming 3B runs"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# D 2-layer: seed 42 done, need 123 and 456
echo ""
echo ">>> Condition D: 2 inserted layers + SFT (seeds 123, 456)"
echo "============================================"
cleanup
python -m experiment.train --condition d --model $MODEL --output-dir $OUTPUT_DIR --seeds 123,456 --insertion-positions 8,18

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition d --model $MODEL --output-dir $OUTPUT_DIR --seeds 42,123,456 --insertion-positions 5,10,18,23

echo ""
echo ">>> Condition G: LoRA distillation (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition g --model $MODEL --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds 42,123,456 --insertion-positions 8,18

echo ""
echo "============================================"
echo "All remaining 3B runs complete!"
echo "Results saved to $OUTPUT_DIR"
echo "============================================"
