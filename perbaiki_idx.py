"""
Bangun ulang file {part}_idx.npy yang RUSAK (mis. terbaca sebagai pickle/objek) TANPA membuka isinya.

Prinsip: indeks train/val/test adalah pembagian baris data.csv (0..N-1). Untuk bagian yang rusak:
  - kandidat baris = semua baris data.csv yang TIDAK ada di indeks bagian lain yang utuh;
  - urutan = urutan baris di {part}.csv (CSV split dibuat dari data.csv.iloc[idx]) -> tiap baris CSV dicocokkan
    dengan baris data.csv di antara kandidat (baris kembar boleh ditukar karena isinya identik).
  - DIVERIFIKASI: data.csv.iloc[idx_baru] harus sama persis dengan {part}.csv; bila tidak, file TIDAK ditulis.
File rusak disimpan sebagai cadangan {part}_idx.npy.rusak sebelum diganti.

Pemakaian (dari folder proyek):
  python perbaiki_idx.py              # cari & perbaiki semua *_idx.npy yang rusak di datasets/
  python perbaiki_idx.py --cek        # hanya tampilkan yang rusak, tidak menulis apa pun
"""
import os
import glob
import shutil
import argparse

import numpy as np
import pandas as pd

PARTS = ['train', 'val', 'test']


def load_ok(path):
    try:
        a = np.load(path, allow_pickle=False)
        return a if (a.ndim == 1 and np.issubdtype(a.dtype, np.integer)) else None
    except Exception:
        return None


def rebuild(data_dir, ds, seed_dir, part, write):
    full = pd.read_csv(f'{data_dir}/{ds}/data.csv')
    split = pd.read_csv(f'{seed_dir}/{part}.csv')
    others = [load_ok(f'{seed_dir}/{p}_idx.npy') for p in PARTS if p != part]
    if any(o is None for o in others):
        return False, 'indeks bagian lain juga rusak -> tidak bisa dibangun ulang otomatis'
    used = np.concatenate(others)
    cand = np.setdiff1d(np.arange(len(full)), used)
    if len(cand) != len(split):
        return False, f'jumlah kandidat {len(cand)} != baris {part}.csv {len(split)}'
    cols = list(split.columns)
    if list(full.columns) != cols:
        full = full[cols]
    key = lambda df: df.astype(str).agg('\x1f'.join, axis=1)          # kunci baris (aman untuk NaN/teks)
    kc = key(full.iloc[cand]).to_numpy()
    ks = key(split).to_numpy()
    pool = {}
    for i, k in zip(cand, kc):
        pool.setdefault(k, []).append(i)
    idx = []
    for k in ks:
        if not pool.get(k):
            return False, 'ada baris CSV yang tidak ditemukan di data.csv'
        idx.append(pool[k].pop(0))
    idx = np.array(idx, dtype=np.int64)
    a = full.iloc[idx].reset_index(drop=True)
    if not key(a).equals(key(split)):
        return False, 'verifikasi gagal (data.csv.iloc[idx] != CSV)'
    if write:
        p = f'{seed_dir}/{part}_idx.npy'
        if os.path.exists(p):
            shutil.copy2(p, p + '.rusak')
        np.save(p, idx)
        if load_ok(p) is None or not np.array_equal(np.load(p), idx):
            return False, 'gagal menulis ulang (cek disk)'
    return True, f'{len(idx)} indeks, cocok persis dengan {part}.csv'


def main():
    ap = argparse.ArgumentParser(description='Bangun ulang *_idx.npy yang rusak')
    ap.add_argument('--data_dir', default='datasets')
    ap.add_argument('--cek', action='store_true', help='hanya tampilkan, tidak menulis')
    a = ap.parse_args()
    n_bad = 0
    for seed_dir in sorted(glob.glob(f'{a.data_dir}/*/splits/seed*')):
        ds = seed_dir.replace('\\', '/').split('/')[-3]
        for part in PARTS:
            p = f'{seed_dir}/{part}_idx.npy'
            if not os.path.exists(f'{seed_dir}/{part}.csv') or load_ok(p) is not None:
                continue
            n_bad += 1
            ok, msg = rebuild(a.data_dir, ds, seed_dir, part, write=not a.cek)
            status = ('DIPERBAIKI' if not a.cek else 'BISA DIPERBAIKI') if ok else 'GAGAL'
            print(f'{status}: {p} -> {msg}')
    print('Tidak ada file indeks rusak.' if n_bad == 0 else f'{n_bad} file indeks rusak diproses.')


if __name__ == '__main__':
    main()
