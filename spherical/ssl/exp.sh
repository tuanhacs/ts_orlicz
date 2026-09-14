export CUDA_VISIBLE_DEVICES=0
## Ours
python3 train_eval.py --method sts_orlicz \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 7 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --n_function exp --p_agg 2 \
    --lr 0.05 --seed 0

python3 train_eval.py --method sts_orlicz \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 5 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --n_function exp_squared --p_agg 2 \
    --lr 0.05 --seed 0

python3 train_eval.py --method sts_orlicz \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 9 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --n_function power --p_agg 2 \
    --lr 0.05 --seed 0