# Run ablation experiments for the toy problem

# Hold label constant
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --calibration-lr 0.0 --seed $seed &
done

wait;

# Don't learn nominal
for seed in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --exclude-nominal --seed $seed &
done
