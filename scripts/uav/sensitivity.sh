# Run sensitivity sweep on K for the UAV problem

for K in 1 2 4 6 8 10; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --seed $seed --project-suffix neurips-sensitivity --balance --regularization-weight 1.0 --n-calibration-permutations $K &
    done

    # if (( K % 4 == 0 )); then
    #     wait;
    # fi
done

# # test with no calibration
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/uav/train.py --seed $seed --project-suffix sensitivity --n-calibration-permutations 1 --calibration-lr 0 &
# done
