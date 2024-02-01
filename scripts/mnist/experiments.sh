# Run experiments for the two moons dataset

# Our approach
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/mnist/train.py --seed $seed &
done

wait;

# # Baseline: KL-regularized normalizing flow)  RUNS OUT OF MEMORY
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
#     CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &
# done
# Baseline: unregularized normalizing flow
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate &
done

# wait;

# # Baseline: W2-regularized normalizing flow (RNODE)
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
#     CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &
# done
