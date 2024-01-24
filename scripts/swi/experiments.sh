# Run experiments for the seismic wave inversion problem

# for seed in 0 1 2 3; do
for seed in 0; do
    # Our approach
    CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed

    # # Baseline: KL-regularized normalizing flow)
    # CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed --no-calibrate --regularize --regularization-weight 1.0
    # CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.1
    # CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.01

    # # Baseline: W2-regularized normalizing flow (RNODE)
    # CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0
    # CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1
    # CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01
done