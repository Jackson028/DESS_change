# # * restaurants
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ./train.py --seed 42 --max_span_size 8 --batch_size 8 --epochs 120 --dataset 14res


