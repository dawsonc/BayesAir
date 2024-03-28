# Run experiments for the seismic wave inversion problem

for seed in 0 1 2 3; do
    # Our approach
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed &
done

wait;

for seed in 0 1 2 3; do
    # Our approach
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --project-suffix classify-opt --gmm --seed $seed &
done

wait;

for seed in 0 1 2 3; do
    # Baseline: KL-regularized normalizing flow)
    CUDA_VISIBLE_DEVICES=0, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=1, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=2, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --regularization-weight 0.01 &

    # wait;

    # Baseline: W2-regularized normalizing flow (RNODE)
    CUDA_VISIBLE_DEVICES=0, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 1.0 &
    CUDA_VISIBLE_DEVICES=1, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.1 &
    CUDA_VISIBLE_DEVICES=2, poetry run python scripts/swi/train.py --project-suffix classify-opt --seed $seed --no-calibrate --regularize --wasserstein --regularization-weight 0.01 &

    wait;
done

wait;