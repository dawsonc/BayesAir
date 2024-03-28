# Run sensitivity sweep on K for the SWI problem

# for K in 1 2 3 4 5 6 7 8 9 10; do
for K in 7 8; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --seed $seed --project-suffix sensitivity --n-calibration-permutations $K &
    done

    # if (( K % 3 == 0 )); then
    #     wait
    # fi
done

# # test with no calibration
# for seed in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/two_moons/train.py --seed $seed --project-suffix sensitivity --n-calibration-permutations 1 --calibration-lr 0 &
# done
