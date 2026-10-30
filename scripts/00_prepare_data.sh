#!/usr/bin/env bash
# Tahap 1–3: dataset, split, mask
set -e

python -m src.data.prepare_datasets
python -m src.data.make_splits
python -m src.data.make_masks
