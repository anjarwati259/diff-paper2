#!/usr/bin/env bash
# E4: 5 varian, MAR 10% & 40%
set -e

for ds in bike news california house_sales miami; do
  for ratio in 0.1 0.4; do
    for seed in 0 1 2 3 4; do
      for varian in diffputer_asli usulan_bobot usulan_masking usulan_sem usulan_lengkap; do
        python -m src.model.run_diffputer --dataset $ds --mekanisme MAR --ratio $ratio --seed $seed --varian $varian
      done
    done
  done
done
