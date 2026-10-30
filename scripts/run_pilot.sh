#!/usr/bin/env bash
# Pilot: 4 varian, California, MAR 10% & 40%, seed 0–2
set -e

for ratio in 0.1 0.4; do
  for seed in 0 1 2; do
    for varian in diffputer_asli usulan_masking usulan_sem usulan_lengkap; do
      python -m src.model.run_diffputer --dataset california --mekanisme MAR --ratio $ratio --seed $seed --varian $varian
    done
  done
done
python -m src.eval.aggregate --filter pilot
