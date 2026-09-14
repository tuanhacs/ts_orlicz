export CUDA_VISIBLE_DEVICES=1

# Table
python3 main.py -d ssw
python3 main.py -d s3w
python3 main.py -d ri_s3w_1
python3 main.py -d ri_s3w_5
python3 main.py -d ari_s3w
python3 main.py -d stsw --p 1 --delta 50
python3 main.py -d stsw_gen --p 1 --delta 1 --lr 0.05
python main.py -d sts_orlicz --delta 1 --lr 0.05 --n_function exp --p_agg 2
python main.py -d sts_orlicz --delta 1 --lr 0.05 --n_function exp_squared --p_agg 2
python main.py -d sts_orlicz --p 2 --delta 1 --lr 0.05 --n_function function --p_agg 2

# Figure
# python3 plot_loss.py