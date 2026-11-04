"""
Cek keutuhan file data & kode (setelah ada file rusak di drive: null byte / file .npy terbaca sebagai pickle).

Memeriksa, untuk setiap dataset di datasets/Info dan setiap seed:
  - datasets/{ds}/data.csv dan splits/seed{k}/{train,val,test}.csv : tanpa null byte, terbaca, kolom lengkap
  - splits/seed{k}/{part}_idx.npy : terbaca TANPA pickle, 1 dimensi, bilangan bulat, panjang = baris CSV
  - masks/{tag}/seed{k}/{part}_mask.npy : terbaca TANPA pickle, bentuk (baris CSV, jumlah fitur)
  - semua *.py di folder proyek & baselines_eksternal/ (atau baseline_eksternal/) : tanpa null byte, sintaks benar

Pemakaian (dari folder proyek):  python cek_data.py            # semua dataset
                                 python cek_data.py --dataset bike
Hasil: daftar RUSAK (bila ada) + ringkasan. File rusak: salin ulang dari cadangan / buat ulang dengan skrip split asli.
"""
import os
import ast
import glob
import json
import argparse

import numpy as np
import pandas as pd

PARTS = ['train', 'val', 'test']


def has_null(path):
    with open(path, 'rb') as f:
        return b'\x00' in f.read()


def main():
    ap = argparse.ArgumentParser(description='Cek keutuhan file data & kode')
    ap.add_argument('--dataset', nargs='*', default=None)
    a = ap.parse_args()
    bad, n_ok = [], 0

    def ok(cond, path, why=''):
        nonlocal n_ok
        if cond:
            n_ok += 1
        else:
            bad.append((path, why))

    # ---- kode ----
    for f in sorted(glob.glob('*.py') + glob.glob('baselines_eksternal/*.py') + glob.glob('baseline_eksternal/*.py')):
        if has_null(f):
            ok(False, f, 'null byte'); continue
        try:
            ast.parse(open(f, encoding='utf-8').read()); ok(True, f)
        except Exception as e:
            ok(False, f, f'sintaks: {e}')

    # ---- data ----
    names = a.dataset or [os.path.basename(p)[:-5] for p in sorted(glob.glob('datasets/Info/*.json'))]
    for ds in names:
        info = json.load(open(f'datasets/Info/{ds}.json'))
        cols = info['num_cols'] + info['cat_cols'] + [info['target']]
        n_feat = len(info['num_cols']) + len(info['cat_cols'])
        p = f'datasets/{ds}/data.csv'
        if os.path.exists(p):
            ok(not has_null(p), p, 'null byte')
        for sd in sorted(glob.glob(f'datasets/{ds}/splits/seed*')):
            seed = os.path.basename(sd)
            nrow = {}
            for part in PARTS:
                c = f'{sd}/{part}.csv'
                try:
                    if has_null(c):
                        raise ValueError('null byte')
                    df = pd.read_csv(c)
                    miss = [k for k in cols if k not in df.columns]
                    if miss:
                        raise ValueError(f'kolom hilang {miss[:3]}')
                    nrow[part] = len(df); ok(True, c)
                except Exception as e:
                    ok(False, c, str(e)[:80])
                ix = f'{sd}/{part}_idx.npy'
                try:
                    v = np.load(ix, allow_pickle=False)
                    good = v.ndim == 1 and np.issubdtype(v.dtype, np.integer) and len(v) == nrow.get(part, len(v))
                    ok(good, ix, f'dtype {v.dtype} bentuk {v.shape}, baris CSV {nrow.get(part)}')
                except Exception as e:
                    ok(False, ix, f'{type(e).__name__}: {str(e)[:60]}')
            for md in sorted(glob.glob(f'datasets/{ds}/masks/*/{seed}')):
                for part in PARTS:
                    m = f'{md}/{part}_mask.npy'
                    if not os.path.exists(m):
                        continue
                    try:
                        v = np.load(m, allow_pickle=False)
                        good = v.shape == (nrow.get(part, v.shape[0]), n_feat)
                        ok(good, m, f'bentuk {v.shape}, harap ({nrow.get(part)}, {n_feat})')
                    except Exception as e:
                        ok(False, m, f'{type(e).__name__}: {str(e)[:60]}')

    print(f'{n_ok} file OK, {len(bad)} file RUSAK')
    for pth, why in bad:
        print(f'  RUSAK: {pth}  ->  {why}')
    if not bad:
        print('Semua file utuh.')


if __name__ == '__main__':
    main()
