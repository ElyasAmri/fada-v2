#!/bin/bash
# Batch rescore all checkpoints with updated scoring pipeline
PYTHON="./venv/Scripts/python.exe"
RESULTS="experiments/rccg/results"
CONVERT="experiments/evaluation/checkpoint_to_predictions.py"
SCORE="experiments/evaluation/score_against_gt.py"

for cp in "$RESULTS"/checkpoint_*.json; do
    basename=$(basename "$cp")
    model=$(echo "$basename" | sed 's/^checkpoint_vllm_//;s/^checkpoint_//;s/\.json$//')
    pred="$RESULTS/predictions_${model}.jsonl"
    score="$RESULTS/scores_${model}.json"

    # Convert if no predictions file
    if [ ! -f "$pred" ]; then
        echo "Converting $model..."
        $PYTHON -u "$CONVERT" --checkpoint "$cp" --output "$pred" 2>&1 | tail -1
    fi

    # Rescore
    echo "Scoring $model..."
    $PYTHON -u "$SCORE" --predictions "$pred" --output "$score" 2>&1 | tail -1

    # Show result
    $PYTHON -c "
import json
d=json.load(open('$score'))
o=d['overall']
print(f'  {o[\"primary_score_mean\"]:.4f} | {o[\"embedding_similarity_mean\"]:.4f}')
" 2>&1
done

echo "=== ALL DONE ==="
