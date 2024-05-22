# Run experiments for the two moons dataset

# Our approach
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix speed_test --seed $seed --balance &
done

# wait;

# Baseline: KL-regularized normalizing flow)
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &
done

wait;

# Baseline: W2-regularized normalizing flow (RNODE)
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &
done

wait;

# Baseline: GMM
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --project-suffix classify-opt --gmm --seed $seed &
done

wait;