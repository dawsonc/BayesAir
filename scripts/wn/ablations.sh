# Run ablation experiments for the ATC problem

# # Ours
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --seed $seed &
# done

# # Hold label constant
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --calibration-lr 0.0 --seed $seed &
# done

# wait;

for seed in 0 1 2 3; do
    # Don't learn nominal
    CUDA_VISIBLE_DEVICES=2, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --calibration-lr 0.0 --exclude-nominal --seed $seed &

    # Don't subsample
    CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --calibration-lr 0.0 --exclude-nominal --no-calibrate --seed $seed &

    wait;
done

# # Don't subsample
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=3, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --calibration-lr 0.0 --exclude-nominal --no-calibrate --seed $seed &
# done
