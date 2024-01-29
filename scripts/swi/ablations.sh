# Run ablation experiments for the seismic wave inversion problem
# # Ours
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/swi/train.py --seed $seed &
# done

# # Hold label constant
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=0, poetry run python scripts/swi/train.py --calibration-lr 0.0 --seed $seed &
# done

# wait;

# Don't learn nominal
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=2, poetry run python scripts/swi/train.py --calibration-lr 0.0 --exclude-nominal --seed $seed &
done

# Don't subsample
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=3, poetry run python scripts/swi/train.py --calibration-lr 0.0 --exclude-nominal --no-calibrate --seed $seed &
done
