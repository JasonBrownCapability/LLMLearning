#!/bin/bash
# Llama 3.2 3B runs to verify pattern holds across model sizes
# Run from the project root: bash run_3b.sh
#
# Estimated total time: ~12 hours

set -e

OUTPUT_DIR="/workspace/results_3b"
MODEL="meta-llama/Llama-3.2-3B"
LORA_PATH="$OUTPUT_DIR/condition_b_lora_rl"
SEEDS="42,123,456"

mkdir -p $OUTPUT_DIR

# Clean caches between runs to prevent disk full
cleanup() { rm -rf /root/.cache/* /tmp/pip-* 2>/dev/null; echo "Cache cleaned"; }

echo "============================================"
echo "Llama 3.2 3B runs"
echo "Model: $MODEL"
echo "Seeds: $SEEDS"
echo "Output: $OUTPUT_DIR"
echo "============================================"

echo ""
echo ">>> Condition A: Baseline (single seed)"
echo "============================================"
cleanup
python -m experiment.train --condition a --model $MODEL --output-dir $OUTPUT_DIR

echo ""
echo ">>> Condition B: LoRA + RL (single seed)"
echo "============================================"
cleanup
python -m experiment.train --condition b --model $MODEL --output-dir $OUTPUT_DIR

echo ""
echo ">>> Condition D: 2 inserted layers + SFT (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition d --model $MODEL --output-dir $OUTPUT_DIR --seeds $SEEDS --insertion-positions 8,18

echo ""
echo ">>> Condition D: 4 inserted layers + SFT (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition d --model $MODEL --output-dir $OUTPUT_DIR --seeds $SEEDS --insertion-positions 5,10,18,23

echo ""
echo ">>> Condition G: LoRA distillation (3 seeds)"
echo "============================================"
cleanup
python -m experiment.train --condition g --model $MODEL --reuse-lora $LORA_PATH --output-dir $OUTPUT_DIR --seeds $SEEDS --insertion-positions 8,18

echo ""
echo "============================================"
echo "All 3B runs complete!"
echo "Results saved to $OUTPUT_DIR"
echo "============================================"
