# Run ablation experiments for the seismic wave inversion problem

# Hold label constant
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --calibration-lr 0.0 --seed $seed &
done

wait;

# Don't learn nominal
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --exclude-nominal --seed $seed &
done
