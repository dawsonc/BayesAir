# Run sensitivity sweep on K for the SWI problem

for K in 2 4 8; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --seed $seed --project-suffix neurips-sensitivity --n-steps 250 --n-calibration-permutations $K --balance &
    done

    if (( K == 4 )); then
        wait;
    fi
done

wait;

for K in 2 4 8; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --seed $seed --project-suffix neurips-sensitivity --n-steps 250 --n-calibration-permutations $K --balance --regularization-weight 0.0 &
    done

    if (( K == 4 )); then
        wait;
    fi
done

wait

for K in 2 4 8; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --seed $seed --project-suffix neurips-sensitivity --n-steps 250 --n-calibration-permutations $K --balance --calibration-weight 0.0 &
    done

    if (( K == 4 )); then
        wait;
    fi
done

wait

for K in 2 4 8; do
    for seed in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$seed, poetry run python scripts/swi/train.py --seed $seed --project-suffix neurips-sensitivity --n-steps 250 --n-calibration-permutations $K --balance --calibration-lr 0.0 &
    done

    if (( K == 4 )); then
        wait;
    fi
done
