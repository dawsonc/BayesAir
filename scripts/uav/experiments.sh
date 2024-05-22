# Run experiments for the uav problem

for seed in 0 1 2 3; do
    # Our approach
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --balance --seed $seed &
done

# wait;

for seed in 0 1 2 3; do
    # Baseline: KL-regularized normalizing flow)
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &
done

for seed in 0 1 2 3; do
    # Baseline: ensemble
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --no-calibrate --bagged --seed $seed &
done

wait;

for seed in 0 1 2 3; do
    # Baseline: W2-regularized normalizing flow (RNODE)
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --n-steps 500 --project-suffix neurips-new --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &
done

# wait;