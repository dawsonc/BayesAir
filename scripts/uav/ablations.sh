# Run ablation experiments for the seismic wave inversion problem
# Ours
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --ablation --seed $seed &
done

# Hold label constant
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --ablation --calibration-lr 0.0 --seed $seed &
done

# Don't learn nominal
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --ablation --calibration-lr 0.0 --exclude-nominal --seed $seed &
done

# Don't subsample
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --ablation --calibration-lr 0.0 --exclude-nominal --no-calibrate --seed $seed &
done

