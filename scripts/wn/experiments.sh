# Run experiments for the ATC problem

for seed in 0 1 2 3; do
    # Our approach
    CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --balance --seed $seed &

    # KL regularized
    CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &

    wait;
done

wait;

for seed in 0 1 2 3; do
    # Ensemble
    CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --bagged &

    # W2 regularized
    CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix neurips --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &
done
