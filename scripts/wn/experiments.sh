# Run experiments for the ATC problem

CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --seed 0

# for seed in 0 1 2 3; do
#     # Our approach
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --seed $seed
# done

# for seed in 0 1 2 3; do
#     # Baseline: KL-regularized normalizing flow)
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --seed $seed --no-calibrate --regularize --regularization-weight 1.0
#     CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.1
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.01

#     # Baseline: W2-regularized normalizing flow (RNODE)
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0
#     CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01
# done