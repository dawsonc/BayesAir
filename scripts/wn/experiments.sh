# Run experiments for the ATC problem

for seed in 0 1 2 3; do
    # Our approach
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed &
done

# wait;

CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 0 --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 0 --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 0 --no-calibrate --regularize --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 1 --no-calibrate --regularize --regularization-weight 1.0 &
wait;
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 1 --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 1 --no-calibrate --regularize --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 2 --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 2 --no-calibrate --regularize --regularization-weight 0.1 &
# wait
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 2 --no-calibrate --regularize --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 3 --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 3 --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 3 --no-calibrate --regularize --regularization-weight 0.01 &

wait;

CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 0 --no-calibrate --regularize  --wasserstein --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 0 --no-calibrate --regularize  --wasserstein --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 0 --no-calibrate --regularize  --wasserstein --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 1 --no-calibrate --regularize  --wasserstein --regularization-weight 1.0 &
# wait;
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 1 --no-calibrate --regularize  --wasserstein --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 1 --no-calibrate --regularize  --wasserstein --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 2 --no-calibrate --regularize  --wasserstein --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 2 --no-calibrate --regularize  --wasserstein --regularization-weight 0.1 &
wait
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 2 --no-calibrate --regularize  --wasserstein --regularization-weight 0.01 &

CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 3 --no-calibrate --regularize  --wasserstein --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 3 --no-calibrate --regularize  --wasserstein --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed 3 --no-calibrate --regularize  --wasserstein --regularization-weight 0.01 &

wait;

# for seed in 0 1 2 3; do
#     # Baseline: KL-regularized normalizing flow)
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
#     CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &

#     # Baseline: W2-regularized normalizing flow (RNODE)
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
#     CUDA_VISIBLE_DEVICES=1, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
#     CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-steps 150 --project-suffix classify-opt-3 --n-failure 4 --n-failure-eval 4 --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &
# done
