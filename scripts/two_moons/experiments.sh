# Run experiments for the two moons dataset

# Our approach; confirmed working
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --calibration-lr 1e-2

# Baseline: beta-NF (KL-regularized normalizing flow); not tested
# Needs testing with a range of weights for the KL normalization
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --no-calibrate --regularize --regularization-weight 1.0
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --no-calibrate --regularize --regularization-weight 0.1
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --no-calibrate --regularize --regularization-weight 0.01
# TODO: add a run with optimal KL regularization weight determined via manual tuning

# Baseline: W2-regularized normalizing flow (RNODE); not tested
# Needs testing with a range of weights for the W2 regularization
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --no-calibrate --regularize --wasserstein --regularization-weight 1.0
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --no-calibrate --regularize --wasserstein --regularization-weight 0.1
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/two_moons/train.py --no-calibrate --regularize --wasserstein --regularization-weight 0.01
# TODO: add a run with optimal W2 regularization weight determined via manual tuning
