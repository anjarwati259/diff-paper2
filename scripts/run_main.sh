#!/usr/bin/env bash
# E1 + E2: semua dataset, MAR 0–40%, seed 0–4, usulan lengkap + DiffPuter asli
set -e

for ds in bike news california house_sales miami; do
  for ratio in 0.0 0.1 0.2 0.3 0.4; do
    for seed in 0 1 2 3 4; do
      for varian in diffputer_asli usulan_lengkap; do
        python -m src.model.run_diffputer --dataset $ds --mekanisme MAR --ratio $ratio --seed $seed --varian $varian
      done
    done
  done
done
