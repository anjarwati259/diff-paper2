"""
Evaluasi ulang HASIL LAMA (metrik skala asli) ke SKALA STANDAR-LOG, tanpa training ulang.

Skala standar-log = z-score dari log1p(y) dengan mean/std y TRAIN (= skala tempat semua model dilatih;
tanpa log bila log_target = false). Sama dengan data_loader.y_true_z / inverse_y versi baru.

Cara kerja: telusuri folder hasil, ambil setiap {nama}_{val|test}.csv berkolom y_asli & y_pred (regresi).
  - Ada sampel {nama}_{part}_samples.npy (usulan, DiffPuter asli, XGBoost kuantil):
      sampel -> log1p -> z; prediksi titik = rata-rata sampel di skala z; RMSE, MAE, R2, CRPS, coverage.
      File per iterasi {nama}_{part}_samples_iter{i}.npy juga dihitung (kolom 'iterasi'), ditambah baris
      '[ensemble iterasi]' = gabungan sampel SEMUA iterasi EM (snapshot ensemble, tanpa pemilihan).
      Coverage min-max ensemble: ideal (N_total - 1)/(N_total + 1), mis. 50 sampel -> 0.961.
      Bila ada {nama}_{part}_pm_ensemble.npy (run_usulan_final.py versi pm) atau {nama}_evalpm_{part}_pm_ensemble.npy
      (eval_ckpt_reg.py): baris '[ensemble pm]' = RMSE/MAE/R2 dari posterior mean ensemble, CRPS/coverage dari gabungan
      sampel. Ini HASIL UTAMA metode usulan regresi.
  - Tanpa sampel (LightGBM, CatBoost, NeuMiss, GRAPE, supMIWAE, dua tahap SVR): y_pred -> log1p -> z;
      RMSE, MAE, R2 saja. (y_pred metode log lama = expm1(prediksi log) -> log1p mengembalikannya persis.)
  - Satu file sampel {nama}_{part}_samples.npy (XGBoost, LightGBM/CatBoost/supMIWAE versi baru): RMSE/MAE/R2 dari
      y_pred (model titik, sama dengan skrip run); CRPS & coverage dari sampel. GBDT: sampel = kuantil k/(N+1)
      -> CRPS rumus kuantil.
Baris CSV dicek cocok dengan split. Hasil LAMA (y_asli = nilai asli) dikonversi ke z-log; hasil BARU dari skrip
versi standar-log (results_zlog/, y_asli sudah z-log) dibaca apa adanya. Selain itu file dilewati.

Pemakaian (dari folder proyek):
  python eval_zlog.py results results_4 results_5 baselines_eksternal/results          # hasil lama
  python eval_zlog.py results_zlog baselines_eksternal/results_zlog --out rekap_baru     # hasil baru
Hasil: eval_zlog_semua.csv (semua file, semua iterasi) dan eval_zlog_ringkas.csv
       (iterasi terakhir, rata-rata ± sd antar seed per metode / dataset / missing / bagian)
"""
import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd

from data_loader import load_split

DS_INFO = 'datasets/Info'


def crps_fair(S, y):
    N = S.shape[0]
    t1 = np.abs(S - y[None]).mean(0)
    t2 = np.abs(S[:, None] - S[None]).sum((0, 1)) / (2 * N * (N - 1))
    return float((t1 - t2).mean())


def crps_quantile(Q, y):
    N = Q.shape[0]
    t1 = np.abs(Q - y[None]).mean(0)
    t2 = np.abs(Q[:, None] - Q[None]).sum((0, 1)) / (2 * N * N)
    return float((t1 - t2).mean())


def point_metrics(y, p):
    return {'rmse': float(np.sqrt(((y - p) ** 2).mean())), 'mae': float(np.abs(y - p).mean()),
            'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def sample_metrics(y, S, quantile):
    """S [N, n] skala z."""
    m = point_metrics(y, S.mean(0))
    lo, hi = np.percentile(S, 5, axis=0), np.percentile(S, 95, axis=0)
    mn, mx = S.min(0), S.max(0)
    m.update({'crps': crps_quantile(np.sort(S, 0), y) if quantile else crps_fair(S, y),
              'coverage_p5_p95': float(((y >= lo) & (y <= hi)).mean()),
              'coverage_minmax': float(((y >= mn) & (y <= mx)).mean()),
              'coverage_minmax_ideal': (S.shape[0] - 1) / (S.shape[0] + 1),
              'width_minmax': float((mx - mn).mean()), 'n_sampel': S.shape[0]})
    return m


_cache = {}


def stats(ds, seed):
    """mean/std y skala model (log1p bila log_target) dari TRAIN; tidak bergantung pada mask."""
    if (ds, seed) not in _cache:
        data, meta = load_split(ds, seed, 0.0)
        i = meta['y_idx']
        _cache[(ds, seed)] = (meta['mean'][i], meta['std'][i], meta['log_target'], meta['target'], data)
    return _cache[(ds, seed)]


def to_z(v, mu, sd, log_t):
    v = np.asarray(v, dtype=np.float64)
    return ((np.log1p(np.clip(v, -0.999999, None)) if log_t else v) - mu) / sd


def main():
    ap = argparse.ArgumentParser(description='Evaluasi ulang hasil lama ke skala standar-log')
    ap.add_argument('roots', nargs='+', help='folder hasil yang ditelusuri')
    ap.add_argument('--out', default='eval_zlog')
    a = ap.parse_args()

    reg_ds = []
    for f in sorted(glob.glob(f'{DS_INFO}/*.json')):
        import json
        if json.load(open(f)).get('task_type', 'regression') == 'regression':
            reg_ds.append(os.path.basename(f)[:-5])
    print(f'dataset regresi: {reg_ds}')

    rows, skip = [], []
    for root in a.roots:
        if not os.path.isdir(root):
            print(f'[PERINGATAN] folder tidak ditemukan: {root}  (cek ejaan, mis. baseline_eksternal vs baselines_eksternal)')
            continue
        n_csv = len(glob.glob(f'{root}/**/*_*.csv', recursive=True))
        print(f'[{root}] {n_csv} file CSV ditemukan')
        for csv in sorted(glob.glob(f'{root}/**/*_*.csv', recursive=True)):
            name = os.path.basename(csv)
            mpart = re.match(r'(.+)_(val|test)\.csv$', name)
            if not mpart:
                continue
            prefix, part = mpart.groups()
            comps = csv.replace('\\', '/').split('/')
            ds = next((c for c in comps if c in reg_ds), None)
            sd_ = next((c for c in comps if re.fullmatch(r'seed\d+', c)), None)
            tag = next((c for c in comps if re.fullmatch(r'(MAR|MCAR)\d+', c)), '')
            if ds is None or sd_ is None:
                continue
            df = pd.read_csv(csv)
            if not {'y_asli', 'y_pred'} <= set(df.columns):
                continue
            seed = int(sd_[4:])
            mu, sd, log_t, target, data = stats(ds, seed)
            y_raw = data[part]['df'][target].to_numpy(np.float64)
            y = to_z(y_raw, mu, sd, log_t)
            ya = df['y_asli'].to_numpy(np.float64)
            if len(df) == len(y_raw) and np.allclose(ya, y_raw, rtol=1e-6, atol=1e-6):
                conv = lambda v: to_z(v, mu, sd, log_t)                         # hasil LAMA: skala asli -> z-log
            elif len(df) == len(y_raw) and np.allclose(ya, y, rtol=1e-5, atol=1e-5):
                conv = lambda v: np.asarray(v, dtype=np.float64)                 # hasil BARU: sudah z-log
            else:
                skip.append((csv, 'y_asli tidak cocok dengan split (seed/rasio/split lain?)'))
                continue
            folder = os.path.dirname(csv)
            metode = (os.path.basename(os.path.normpath(root)) + '/' +
                      os.path.relpath(folder, root).replace('\\', '/').split('/')[0] + ' | ' + prefix)
            base = {'root': root, 'metode': metode, 'file': prefix, 'dataset': ds, 'missing': tag,
                    'seed': seed, 'bagian': part}
            quantile = any(k in prefix for k in ('xgboost', 'lightgbm', 'catboost'))   # sampel = kuantil
            samp = f'{folder}/{prefix}_{part}_samples.npy'
            iters = sorted(glob.glob(f'{folder}/{prefix}_{part}_samples_iter*.npy'),
                           key=lambda p: int(re.search(r'iter(\d+)\.npy$', p).group(1)))
            if iters:
                allS = []
                for p in iters:
                    S = conv(np.load(p).T)                       # file [n, N] -> [N, n]
                    it = int(re.search(r'iter(\d+)\.npy$', p).group(1))
                    rows.append({**base, 'iterasi': it, 'sumber': 'sampel', **sample_metrics(y, S, quantile)})
                    allS.append(S)
                if len(allS) > 1:      # ensemble antar iterasi EM: gabungan sampel SEMUA iterasi (tanpa pemilihan)
                    rows.append({**base, 'metode': metode + ' [ensemble iterasi]', 'iterasi': -1,
                                 'sumber': f'sampel iterasi 0-{len(allS) - 1}',
                                 **sample_metrics(y, np.concatenate(allS, 0), quantile)})
                # ensemble POSTERIOR MEAN (run_usulan_final.py baru atau eval_ckpt_reg.py): titik dari pm,
                # CRPS/coverage dari gabungan sampel semua iterasi. File pm sudah skala standar-log.
                for pmf in (f'{folder}/{prefix}_{part}_pm_ensemble.npy', f'{folder}/{prefix}_evalpm_{part}_pm_ensemble.npy'):
                    if os.path.exists(pmf):
                        pm = np.load(pmf).astype(np.float64)
                        if len(pm) == len(y):
                            m = sample_metrics(y, np.concatenate(allS, 0), quantile)
                            m.update(point_metrics(y, pm))
                            rows.append({**base, 'metode': metode + ' [ensemble pm]', 'iterasi': -1,
                                         'sumber': f'pm {os.path.basename(pmf)} + sampel iterasi 0-{len(allS) - 1}', **m})
                        break
            elif os.path.exists(samp):
                S = conv(np.load(samp).T)
                m = sample_metrics(y, S, quantile)
                m.update(point_metrics(y, conv(df['y_pred'])))   # prediksi titik = y_pred (model titik, sama dgn skrip run)
                rows.append({**base, 'iterasi': -1, 'sumber': 'sampel', **m})
            else:
                rows.append({**base, 'iterasi': -1, 'sumber': 'y_pred CSV',
                             **point_metrics(y, conv(df['y_pred']))})

    if not rows:
        for c, why in skip[:20]:
            print(f'  dilewati: {c} -> {why}')
        sys.exit('Tidak ada file hasil regresi yang cocok (lihat daftar di atas).')
    R = pd.DataFrame(rows)
    R.to_csv(f'{a.out}_semua.csv', index=False)
    last = R.loc[R.groupby(['root', 'metode', 'dataset', 'missing', 'seed', 'bagian'])['iterasi'].idxmax()]
    cols = [c for c in ['rmse', 'mae', 'r2', 'crps', 'coverage_minmax'] if c in last]
    g = last.groupby(['dataset', 'missing', 'bagian', 'metode'])[cols]
    ring = g.mean().add_suffix('_mean').join(g.std().add_suffix('_sd')).join(g.size().rename('n_seed'))
    ring = ring.reset_index().sort_values(['dataset', 'missing', 'bagian', 'rmse_mean'])
    ring.to_csv(f'{a.out}_ringkas.csv', index=False)
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)
    for (ds, ms, pt), t in ring.groupby(['dataset', 'missing', 'bagian']):
        print(f'\n=== {ds} {ms} {pt} — skala standar-log, iterasi terakhir, rata-rata antar seed ===')
        print(t[['metode', 'n_seed'] + [f'{c}_mean' for c in cols]].to_string(index=False, float_format='%.4f'))
    if skip:
        print(f'\n{len(skip)} file dilewati, contoh: {skip[:3]}')
    print(f'\nTersimpan: {a.out}_semua.csv, {a.out}_ringkas.csv')


if __name__ == '__main__':
    main()