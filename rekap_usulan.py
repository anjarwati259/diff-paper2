"""
Rekap hasil metode_usulan.py: rata-rata dan per seed, untuk setiap dataset, rasio, varian dan cara baca.

    python rekap_usulan.py                          # semua hasil di results_zlog/usulan
    python rekap_usulan.py --part val               # bagian val (default test)
    python rekap_usulan.py --dataset bike --out rekap_bike.csv

Metrik: regresi RMSE, MAE, R2, CRPS; klasifikasi Acc, F1 macro, AUC (sama dengan yang dicetak run).
Cara baca: pm_terakhir (metode final), sampel_terakhir (cara DiffPuter), pm_ensemble, sampel_ensemble.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRIK = {'regression': ['rmse', 'mae', 'r2', 'crps'], 'classification': ['acc', 'f1_macro', 'auc']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results_zlog/usulan')
    ap.add_argument('--part', default='test', choices=['val', 'test'])
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--out', default=None, help='simpan tabel rata-rata ke CSV')
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f'{a.root}/*/*/seed*/*_metrics.json')):
        js = json.load(open(f))
        if 'hasil' not in js or (a.dataset and js['dataset'] != a.dataset):
            continue                                            # run belum selesai / dataset lain
        tag = os.path.basename(os.path.dirname(os.path.dirname(f)))
        for baca, isi in js['hasil'].items():
            if a.part not in isi:
                continue
            rows.append({'dataset': js['dataset'], 'tag': tag, 'varian': js['varian'], 'task': js['task'],
                         'seed': js['seed'], 'cara_baca': baca,
                         'utama': baca == f"{js['utama']['readout']}_{js['utama']['hasil']}", **isi[a.part]})
    if not rows:
        raise SystemExit(f'Tidak ada hasil selesai di {a.root}')
    df = pd.DataFrame(rows)

    ringkas = []
    for (ds, tag, var, task, baca), g in df.groupby(['dataset', 'tag', 'varian', 'task', 'cara_baca']):
        g = g.sort_values('seed')
        r = {'dataset': ds, 'tag': tag, 'varian': var, 'cara_baca': baca, 'n_seed': len(g),
             'seed': ' '.join(map(str, g.seed))}
        for m in METRIK[task]:
            r[m] = round(float(g[m].mean()), 4)
            r[f'{m}_sd'] = round(float(g[m].std(ddof=1)), 4) if len(g) > 1 else np.nan
            r[f'{m}_per_seed'] = ' '.join(f'{v:.4f}' for v in g[m])
        ringkas.append(r)
    out = pd.DataFrame(ringkas)

    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_colwidth', 60)
    for (ds, tag), g in out.groupby(['dataset', 'tag']):
        task = df[(df.dataset == ds)].task.iloc[0]
        print(f'\n===== {ds} {tag} | {a.part} | {task} =====')
        kol = ['varian', 'cara_baca', 'n_seed'] + [x for m in METRIK[task] for x in (m, f'{m}_sd')]
        print(g[kol].to_string(index=False))
        print('per seed:')
        for _, r in g.iterrows():
            print(f"  {r.varian} | {r.cara_baca} | seed {r.seed} | "
                  + ' | '.join(f'{m} {r[f"{m}_per_seed"]}' for m in METRIK[task]))
    if a.out:
        out.to_csv(a.out, index=False)
        print(f'\nTersimpan: {a.out}')


if __name__ == '__main__':
    main()
