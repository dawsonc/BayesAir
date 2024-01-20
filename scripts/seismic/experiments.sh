CUDA_VISIBLE_DEVICES=0 poetry run python scripts/seismic/train.py &  # ours
CUDA_VISIBLE_DEVICES=0, poetry run python scripts/seismic/train.py --no-calibrate --regularize --regularization-weight 1.0 &
CUDA_VISIBLE_DEVICES=1, poetry run python scripts/seismic/train.py --no-calibrate --regularize --regularization-weight 0.1 &
CUDA_VISIBLE_DEVICES=3, poetry run python scripts/seismic/train.py --no-calibrate --regularize --regularization-weight 0.01 &
