# Run ablation experiments for the ATC problem

# Ours
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --seed $seed &
done

# # Hold label constant
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --calibration-lr 0.0 --seed $seed &
# done

# wait;

# # Don't learn nominal
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/wn/train.py --top-n 4 --n-failure 4 --n-failure-eval 4 --exclude-nominal --seed $seed &
# done
