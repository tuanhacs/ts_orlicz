# Euclidean gradient flow

`GradientFlow.py` optimizes a source point cloud toward a synthetic target and records the squared 2-Wasserstein evaluation distance every 500 iterations. Run it from the **repository root** after following the [installation instructions](../../README.md#environment). The target distributions are generated in `core/utils_GF.py`; no dataset download is needed.

## Run

The script runs every method in this order:

| Index | Result title | Loss branch |
| --- | --- | --- |
| 0 | `TS_Orlicz` | TS-Orlicz with distance-based mass division |
| 1 | `TSW-SL-distance-based` | Tree-sliced Wasserstein with distance-based mass division |
| 2 | `TSW-SL-uniform` | Tree-sliced Wasserstein with uniform mass division |
| 3 | `NonlinearTWD` | Tree-sliced Wasserstein with the selected `--ftype` mapping |
| 4 | `LCVSW` | LCVSW |
| 5 | `SWGG` | SWGG |
| 6 | `MaxSW` | MaxSW |
| 7 | `SW` | Sliced Wasserstein |

Example for the 25 Gaussians target:

```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name 25gaussians --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 \
  --delta 5 --p_tsw 1 --p_sobolev 1.1 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function power --p_agg 1.1 --std 0.1
```
```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name 25gaussians --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 \
  --delta 5 --p_tsw 1 --p_sobolev 1.1 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function exp --std 0.1
```
```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name 25gaussians --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.005 \
  --delta 5 --p_tsw 1 --p_sobolev 1.1 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function exp_squared --std 0.1
```
For the Gaussian 30d target, use `--dataset_name gaussian_30d_small_v --lr_tsw_sl 0.05`. Each run creates 500 source and target samples. Set `--n_function power --p_sobolev 2` for the quadratic TS-Orlicz variant, or select `--n_function exp` or `--n_function exp_squared` for the exponential variants.

`--L` is the total number of lines; the tree methods draw `L / n_lines` trees. `--lr_tsw_sl` is used for the first four methods and `--lr_sw` for the last four. `--ftype` controls `NonlinearTWD` and accepts `linear` (the default), `poly`, `pow`, `circular`, and `circular_r0`; `--degree`, `--pow_beta`, and `--radius` configure those mappings. `--i_max` controls the TS-Orlicz root solver's iteration count. Run `python experiments/gradient-flow/GradientFlow.py --help` for all arguments.

Example for the Gaussian 30d target:

```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name gaussian_30d_small_v --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.05 \
  --delta 5 --p_tsw 1 --p_sobolev 2 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function power --p_agg 2 --std 0.1
```
```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name gaussian_30d_small_v --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.05 \
  --delta 5 --p_tsw 1 --p_sobolev 2 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function exp --std 0.1
```
```bash
python experiments/gradient-flow/GradientFlow.py \
  --dataset_name gaussian_30d_small_v --num_iter 2500 --num_seeds 5 \
  --L 100 --n_lines 4 --lr_sw 0.005 --lr_tsw_sl 0.05 \
  --delta 5 --p_tsw 1 --p_sobolev 2 \
  --ftype pow --degree 3 --pow_beta 1 \
  --n_function exp_squared --std 0.1
```
**Current driver constraints:** Pass an integer `--num_seeds` explicitly; its current default is `0.1`. Keep `--num_iter 2500` because the result-writing block indexes iteration 2500. The tree generation calls use `device='cuda'`, so this script needs a CUDA GPU. The `--plot` and `--eval_sb` flags are currently parsed but do not change the active driver.

The script appends five evaluation values per method to `logs/<title>_resutls.txt` relative to the repository root.
