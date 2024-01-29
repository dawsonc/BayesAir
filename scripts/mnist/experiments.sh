# Run experiments for the two moons dataset

# # Our approach
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/mnist/train.py --seed $seed &
# done

# wait;

# Baseline: KL-regularized normalizing flow)
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
#     CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &
# done
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed 0 --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/mnist/train.py --seed 0 --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed 0 --no-calibrate --regularize --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed 1 --no-calibrate --regularize --regularization-weight 1.0 &
wait;
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed 1 --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/mnist/train.py --seed 1 --no-calibrate --regularize --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed 2 --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed 2 --no-calibrate --regularize --regularization-weight 0.1 &
wait;
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed 2 --no-calibrate --regularize --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=1, poetry run python scripts/mnist/train.py --seed 3 --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed 3 --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed 3 --no-calibrate --regularize --regularization-weight 0.01 &

wait;

# Baseline: W2-regularized normalizing flow (RNODE)
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
#     CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &
# done
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed 0 --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/mnist/train.py --seed 0 --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed 0 --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed 1 --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
wait;
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed 1 --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/mnist/train.py --seed 1 --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed 2 --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed 2 --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
wait;
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/mnist/train.py --seed 2 --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=1, poetry run python scripts/mnist/train.py --seed 3 --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/mnist/train.py --seed 3 --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/mnist/train.py --seed 3 --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &