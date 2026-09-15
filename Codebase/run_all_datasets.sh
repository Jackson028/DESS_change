#!/bin/bash

export TOKENIZERS_PARALLELISM=false

echo "=== DESS Baseline Experiment (All 4 Datasets) ==="
echo "Configuration: DeBERTa-v3-large, batch_size=8, epochs=150, seed=42, max_span_size=8"
echo "Note: Polarity contrastive loss feature available but disabled by default"
echo ""

# 14res
echo "Running 14res..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ./train.py \
    --seed 42 \
    --max_span_size 8 \
    --batch_size 8 \
    --lr 5e-6 \
    --epochs 150 \
    --dataset 14res

# 14lap
echo "Running 14lap..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ./train.py \
    --seed 42 \
    --max_span_size 8 \
    --batch_size 8 \
    --lr 5e-6 \
    --epochs 150 \
    --dataset 14lap

# 15res
echo "Running 15res..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ./train.py \
    --seed 42 \
    --max_span_size 8 \
    --batch_size 8 \
    --lr 1e-5 \
    --epochs 150 \
    --dataset 15res

# 16res
echo "Running 16res..."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ./train.py \
    --seed 42 \
    --max_span_size 8 \
    --batch_size 8 \
    --lr 1e-5 \
    --epochs 150 \
    --dataset 16res

echo ""
echo "=== All Experiments Completed ==="
echo "Results saved in log directory"
