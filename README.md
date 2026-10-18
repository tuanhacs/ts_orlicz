# Tree-Sliced Orlicz experiments

This repository contains Euclidean and spherical tree-sliced Orlicz implementations and experiment code. Start each command block from the repository root; outputs are written relative to the working directory used for that experiment.

## Environment

The supplied [`environment.yaml`](environment.yaml) describes a Linux, Python 3.9, CUDA 11.8 environment. It contains a machine-specific `prefix` line; remove that line before creating an environment on another machine. From the repository root:

```bash
conda env create -f environment.yaml
conda activate twd
python -m pip install -r requirements.txt
python -m pip install -e .
```

Use a CUDA-capable GPU for Euclidean gradient flow, diffusion, and self-supervised learning. Diffusion has additional dependencies in its [README](experiments/denoising-diffusion-gan/README.md).

## Euclidean gradient flow

Run from the **repository root**. The current driver runs eight methods in sequence: TS-Orlicz, 3 tree-sliced Wasserstein variants, LCVSW, SWGG, MaxSW, and SW. Pass `--num_seeds` explicitly; the current default in `cfg.py` is not an integer. Use `--num_iter 2500` because the result writer reads values at iterations 500, 1000, 1500, 2000, and 2500.

```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name 25gaussians --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 \
  --delta 5.0 --p_tsw 1 --p_sobolev 1.1 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function power --p_agg 1.1 --std 0.1
```

For the 30-dimensional Gaussian dataset, use `--dataset_name gaussian_30d_small_v --lr_tsw_sl 0.05`. Other TS-Orlicz choices include `--n_function power --p_sobolev 2`, `--n_function exp`, and `--n_function exp_squared`. The `--ftype`, `--degree`, `--radius`, and `--pow_beta` options configure NonlinearTWD; for example, `--ftype pow --degree 3 --pow_beta 1` selects a nonlinear map.

The driver appends evaluations to `logs/<method>_resutls.txt` and creates directories under `Results_reduced/`. See the [Euclidean gradient-flow README](experiments/gradient-flow/README.md) for method order and argument details.

## Spherical gradient flow

Run from `spherical/gradient_flow` so its relative imports and output paths resolve correctly:

```bash
cd spherical/gradient_flow
python main.py --d_func sts_orlicz --n_function power --p 2 \
  --p_agg 2 --delta 1 --lr 0.05
python main.py --d_func sts_orlicz --n_function exp --p 2 \
  --p_agg 2 --delta 1 --lr 0.05
python main.py --d_func sts_orlicz --n_function exp_squared --p 2 \
  --p_agg 2 --delta 1 --lr 0.05 
```

Use `--n_function exp` or `--n_function exp_squared` for the other Orlicz variants. Select a baseline with `--d_func` (for example, `stsw`, `ssw`, or `s3w`). The script appends results to `results.csv` and `all_resutls.txt` in this directory. Keep `--epochs 500`: the result writer expects measurements through epoch 500.

## Self-supervised learning on CIFAR-10

Run from `spherical/ssl`. The pretraining loader uses `download=False`, so download the CIFAR-10 training and test sets first:

```bash
cd spherical/ssl
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', train=True, download=True); CIFAR10('./data', train=False, download=True)"
python train_eval.py --method sts_orlicz \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 9 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0 --p 2 --n_function power --p_agg 2

python train_eval.py --method sts_orlicz \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 7 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0 --p 2 --n_function exp --p_agg 2

python train_eval.py --method sts_orlicz \
    --ntrees 200 --nlines 20 --delta 2 \
    --unif_w 5 --feat_dim 10 --epochs 200 \
    --batch_size 512 --momentum 0.9 --weight_decay 1e-3 \
    --lr 0.05 --seed 0 --p 2 --n_function exp_squared --p_agg 2
```

`train_eval.py` trains an encoder and runs linear evaluation. It writes checkpoints and metrics under `spherical/ssl/results/`. This pipeline requires a CUDA GPU; the current default selects GPU 0. You can also use `--n_function exp` or `--n_function exp_squared`.

## CIFAR-10 diffusion

Install the additional dependencies described in the [diffusion experiment README](experiments/denoising-diffusion-gan/README.md). From the repository root, this example starts a single-GPU TS-Orlicz training run with W&B in offline mode:

```bash
cd experiments/denoising-diffusion-gan
PYTHONPATH=../.. WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 train_ddgan.py \
  --dataset cifar10 --exp ts_orlicz_exp --loss ts_orlicz \
  --num_channels 3 --num_channels_dae 128 --num_timesteps 4 \
  --num_res_blocks 2 --batch_size 256 --num_epoch 1800 \
  --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 \
  --embedding_type positional --use_ema --ema_decay 0.9999 \
  --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
  --T 2500 --L 4 --twd_delta 10 --twd_std 0.1 \
  --twd_gen_mode gaussian_raw --n_function exp --p_agg 2 --closed_form \
  --ch_mult 1 2 2 2 --save_content --save_ckpt_every 25
```

With `--closed_form`, `exp` and `exp_squared` use simple Taylor approximations of the optimized distance. Omit the flag to optimize the original exponential N-function with the root solver. Set `--n_function power --orlicz_p 2` for the exact quadratic variant. Training downloads CIFAR-10 and writes checkpoints under `saved_info/dd_gan/cifar10/` relative to the diffusion directory. The diffusion README also documents FID evaluation with `test_ddgan.py`.

## Point-cloud gradient flow

Run from the **repository root**. The ShapeNet point-cloud array used by the script is included at `point-cloud-gf/reconstruct_random_50_shapenetcore55.npy`. The `--losses tsw` setting runs the tree-sliced baseline:

```bash
PYTHONPATH=$(pwd) python ./point-cloud-gf/main_point.py \
  --losses ts_orlicz --n_function power --p 1.1 --p_agg 1.1 \
  --data ./point-cloud-gf/reconstruct_random_50_shapenetcore55.npy

PYTHONPATH=$(pwd) python ./point-cloud-gf/main_point.py \
  --losses tsw  \
  --data ./point-cloud-gf/reconstruct_random_50_shapenetcore55.npy
```

For TS-Orlicz, its function can be selected with `--n_function power --p 2`, `--n_function exp`, or `--n_function exp_squared`. By default, `main_point.py` runs five seeds for 500 optimization steps, using GPU 0 if available and CPU otherwise.
