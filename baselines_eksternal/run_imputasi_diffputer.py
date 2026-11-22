"""
BASELINE DUA TAHAP — TAHAP 1 KHUSUS DIFFPUTER: imputasi fitur dengan DiffPuter (Zhang dkk., ICLR 2025),
cara klasik tanpa kebocoran val/test. Tahap 2 (prediksi SVM): run_svm.py --imputer diffputer.

ALUR DIFFPUTER (persis main.py + dataset.py + model.py + diffusion_utils.py resmi; fungsi model, loss, dan
sampling diambil dari run_diffputer.py yang isinya salinan kode resmi):
  Data: fitur numerik + kategorik sebagai bit biner (urutan kategori = kemunculan pertama di data.csv), TANPA
        kolom target (sama dengan dataset.py). Normalisasi mean_std() di nilai teramati train, X = (x-mean)/std/2.
  EM sebanyak max_iter iterasi (bawaan 10 = main.py) di FITUR TRAIN saja:
    M-step : latih model diffusion (EDM, MLP 1024) pada data train yang sudah terisi (iterasi 0: sel hilang = 0)
    E-step : isi sel hilang train = rata-rata num_trials sampel (in-sample); input sampling selalu data asli
             dengan sel hilang = 0, hasil rata-rata menjadi data latih M-step berikutnya
  Setelah iterasi terakhir, val dan test diisi dengan model terakhir (out-of-sample): tiap baris diisi dari
  nilai teramati baris itu sendiri; baris val/test tidak pernah dipakai untuk melatih.
  Setting bawaan = main.py: max_iter 10, max_epochs 10001, patience 500, batch 4096, Adam lr 1e-4,
  ReduceLROnPlateau(0.9, 50), hid_dim 1024, num_trials 20, num_steps 50, resample 20, TANPA clip hasil.
  Satu-satunya tambahan: val juga diimputasi (main.py hanya train & test) agar SVM punya data val.

ATURAN ANTI-KEBOCORAN (sama dengan run_imputasi.py)
  - Kolom target (y) TIDAK ikut: DiffPuter hanya melihat fitur.
  - Standardisasi, kode biner kategorik, dan batas clip dihitung dari TRAIN (data_loader.load_split).
  - Val/test hanya diimputasi, tidak dipakai melatih.

REPRESENTASI
  Ruang asli DiffPuter (data_loader): numerik terstandardisasi + kategorik sebagai bit biner. Setelah imputasi
  dikembalikan ke NILAI ASLI: numerik dibulatkan ke jumlah desimal kolom aslinya (dari train; bilangan bulat ->
  0 desimal), kategorik = kategori valid dengan kode biner terdekat -> label asli. Sel teramati tidak diubah.

KELUARAN (format sama dengan run_imputasi.py, jadi langsung bisa dibaca run_svm.py)
  results/imputasi/diffputer/{dataset}/MAR{rr}/seed{k}/
    missing_{train,val,test}.csv, imputed_{train,val,test}.csv, imputasi_metrics.json, result.txt
  checkpoint: ckpt/twostage_diffputer/{dataset}/MAR{rr}/seed{k}/  (di folder proyek)

Pemakaian (dari folder baselines_eksternal):
  python run_imputasi_diffputer.py                         # semua: DAFTAR_DATASET × DAFTAR_SEED (atur di kode)
  python run_imputasi_diffputer.py --dataset bike --seed 0 # satu run
  lalu: python run_svm.py --imputer diffputer --dataset bike --ratio 0.3 --seed 0
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = next((d for d in (_HERE, os.path.dirname(_HERE)) if os.path.isfile(os.path.join(d, 'data_loader.py'))), None)
if PROJECT_DIR is None:
    raise FileNotFoundError('data_loader.py tidak ditemukan di folder skrip maupun folder induknya')
sys.path.insert(0, _HERE)
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from data_loader import load_split, inverse_transform                       # noqa: E402
from run_imputasi import n_decimals, PARTS                                  # noqa: E402

# =============================================================================================
# DAFTAR YANG DIJALANKAN — ubah di sini (atau timpa lewat argumen --dataset / --seed)
# =============================================================================================
DAFTAR_DATASET = ['bike', 'shoppers']
DAFTAR_SEED = [0, 1, 2, 3, 4]
RASIO = 0.3


def quality(df_imp, df_true, mask_feat, num_cols, cat_cols, std):
    """MAE/RMSE numerik dalam satuan std train (= skala get_eval resmi: (x-mean)/std) dan ACC kategorik."""
    num_err, hits = [], []
    for j, c in enumerate(num_cols):
        m = mask_feat[:, j]
        if m.any():
            num_err.append((df_imp[c].to_numpy(np.float64)[m] - df_true[c].to_numpy(np.float64)[m]) / std[c])
    for j, c in enumerate(cat_cols):
        m = mask_feat[:, len(num_cols) + j]
        if m.any():
            hits.append(df_imp[c].astype(str).to_numpy()[m] == df_true[c].astype(str).to_numpy()[m])
    e = np.concatenate(num_err) if num_err else np.array([np.nan])
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())),
            'acc': float(np.concatenate(hits).mean()) if hits else float('nan'),
            'n_sel_numerik': int(sum(len(x) for x in num_err)), 'n_sel_kategorik': int(sum(len(x) for x in hits))}


def run_one(dataset, seed, a):
    import torch
    from run_diffputer import train_mstep, run_trials
    t_all = time.time()
    dev = f'cuda:{a.gpu}' if torch.cuda.is_available() and a.gpu >= 0 else 'cpu'
    torch.manual_seed(seed); np.random.seed(seed)
    data, meta = load_split(dataset, seed, a.ratio, a.mech)
    num_cols, cat_cols, target = list(meta['num_cols']), list(meta['cat_cols']), meta['target']
    feat = num_cols + cat_cols
    yi = meta['y_idx']                                                     # kolom y dibuang: hanya fitur
    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    out_dir = f'{a.out_dir}/imputasi/diffputer/{dataset}/{tag}/seed{seed}'
    ckpt = f'ckpt/twostage_diffputer/{dataset}/{tag}/seed{seed}'
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== imputasi diffputer | {dataset} ({meta["task"]}) | {tag} seed{seed} | device {dev} | '
          f'missing train {data["train"]["mask_feat"].mean():.3f} ===', flush=True)

    dp = argparse.Namespace(max_epochs=a.max_epochs, patience=a.patience, batch_size=4096, lr=1e-4, hid_dim=1024,
                            num_steps=50, resample=20, num_trials=a.num_trials, chunk_rows=50000)
    bounds = (meta['clip_lo'][:yi], meta['clip_hi'][:yi]) if a.clip else None      # main.py: tanpa clip
    X = {p: data[p]['X_obs'][:, :yi].astype(np.float32) for p in PARTS}   # sel hilang = 0 (= mean train)
    M = {p: data[p]['mask'][:, :yi].astype(np.float32) for p in PARTS}

    # ---- EM di TRAIN saja ----
    t0 = time.time()
    X_miss, rec = X['train'].copy(), {}
    for it in range(a.max_iter):
        os.makedirs(f'{ckpt}/{it}', exist_ok=True)
        print(f'--- iterasi EM {it + 1}/{a.max_iter} ---', flush=True)
        net = train_mstep(X_miss, X_miss.shape[1], dp, dev, f'{ckpt}/{it}', print_model=(it == 0))
        if M['train'].any():
            X_miss = run_trials(net, X['train'], M['train'], dp, dev, 'Imputasi train (in-sample)', bounds).mean(0)
        else:
            X_miss = X['train'].copy()
    rec['train'] = X_miss
    # ---- val & test: hanya diimputasi dengan model terakhir ----
    for p in ('val', 'test'):
        rec[p] = run_trials(net, X[p], M[p], dp, dev, f'Imputasi {p} (out-of-sample)', bounds).mean(0) \
            if M[p].any() else X[p]
    t_imp = time.time() - t0

    # ---- kembali ke NILAI ASLI ----
    miss = {p: data[p]['df'][feat].mask(np.asarray(data[p]['mask_feat'], dtype=bool)).reset_index(drop=True)
            for p in PARTS}
    dtypes = data['train']['df'][feat].dtypes.to_dict()
    decimals = {c: n_decimals(miss['train'][c]) if not pd.api.types.is_integer_dtype(dtypes[c]) else 0 for c in num_cols}
    std_tr = {c: float(meta['std'][j]) for j, c in enumerate(num_cols)}   # std mean_std() resmi (populasi)
    summary = {'imputer': 'diffputer', 'dataset': dataset, 'task': meta['task'], 'mech': a.mech, 'ratio': a.ratio,
               'seed': seed, 'aturan': 'EM dilatih di fitur train saja; val/test hanya diimputasi; y tidak dipakai',
               'setting': {**vars(dp), 'max_iter': a.max_iter, 'clip': a.clip}, 'imputasi_time_s': t_imp,
               'kolom_numerik': num_cols, 'kolom_kategorik': cat_cols, 'desimal_per_kolom': decimals}
    for p in PARTS:
        full = np.concatenate([np.where(M[p] == 1, rec[p], X[p]), data[p]['X_true'][:, yi:]], 1)
        dec = inverse_transform(full, meta)                               # numerik asli + kategori terdekat
        df_true = data[p]['df'][feat].reset_index(drop=True)
        df_imp = pd.DataFrame(index=range(len(df_true)))
        for c in num_cols:
            v = np.round(dec[c].to_numpy(np.float64), decimals[c])
            df_imp[c] = v.astype(dtypes[c]) if pd.api.types.is_integer_dtype(dtypes[c]) else v
        for c in cat_cols:
            col = dec[c]
            if pd.api.types.is_integer_dtype(dtypes[c]):
                col = pd.to_numeric(col, errors='coerce').round()
            df_imp[c] = col.astype(dtypes[c])
        obs = ~np.asarray(data[p]['mask_feat'], dtype=bool)
        for j, c in enumerate(feat):                                      # sel teramati = nilai asli persis
            df_imp.loc[obs[:, j], c] = df_true.loc[obs[:, j], c]
        assert not df_imp.isna().any().any(), f'masih ada sel kosong di {p}'
        df_imp[target] = data[p]['df'][target].to_numpy()
        m_out = miss[p].copy(); m_out[target] = data[p]['df'][target].to_numpy()
        m_out.to_csv(f'{out_dir}/missing_{p}.csv', index=False)
        df_imp.to_csv(f'{out_dir}/imputed_{p}.csv', index=False)
        q = quality(df_imp, df_true, np.asarray(data[p]['mask_feat'], dtype=bool), num_cols, cat_cols, std_tr)
        summary[p] = q
        print(f'Imputasi ({p}) → MAE={q["mae"]:.4f}  RMSE={q["rmse"]:.4f}  ACC={q["acc"]:.4f}', flush=True)
    summary['total_time_s'] = time.time() - t_all
    with open(f'{out_dir}/imputasi_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    te = summary['test']
    with open(f'{out_dir}/result.txt', 'a') as f:
        f.write(f"test | imputasi diffputer | MAE {te['mae']:.4f} RMSE {te['rmse']:.4f} ACC {te['acc']:.4f} | "
                f"waktu {summary['total_time_s']:.0f}s\n")
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {out_dir}/', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Tahap 1 baseline dua tahap: imputasi fitur dengan DiffPuter (fit di train)')
    ap.add_argument('--dataset', nargs='+', default=DAFTAR_DATASET, help='bawaan: DAFTAR_DATASET di kode')
    ap.add_argument('--seed', nargs='+', type=int, default=DAFTAR_SEED, help='bawaan: DAFTAR_SEED di kode')
    ap.add_argument('--ratio', type=float, default=RASIO)
    ap.add_argument('--mech', default='MAR')
    ap.add_argument('--gpu', type=int, default=0, help='indeks GPU (-1 = CPU)')
    ap.add_argument('--max_iter', type=int, default=5, help='iterasi EM (main.py: 10)')
    ap.add_argument('--max_epochs', type=int, default=10001, help='maks epoch per M-step (main.py: 10001)')
    ap.add_argument('--patience', type=int, default=500, help='early stopping M-step (main.py: 500)')
    ap.add_argument('--num_trials', type=int, default=10, help='sampel per imputasi (main.py: 20)')
    ap.add_argument('--clip', action='store_true',
                    help='potong hasil generate ke rentang train ±5%% (TIDAK ada di main.py; bawaan mati)')
    ap.add_argument('--ulang', action='store_true', help='jalankan ulang walau hasilnya sudah ada')
    ap.add_argument('--quick', action='store_true', help='uji alur cepat (hasil TIDAK untuk dilaporkan)')
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'results'),
                    help='folder hasil (bawaan: results/ di folder tempat skrip ini berada)')
    a = ap.parse_args()
    if a.quick:
        a.max_iter, a.max_epochs, a.patience, a.num_trials = 1, 20, 5, 2

    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    jobs = [(d, s) for d in a.dataset for s in a.seed]
    print(f'Total {len(jobs)} run DiffPuter: dataset {a.dataset} × seed {a.seed} | {tag}', flush=True)
    selesai, dilewati, gagal = [], [], []
    for k, (d, s) in enumerate(jobs, 1):
        done = f'{a.out_dir}/imputasi/diffputer/{d}/{tag}/seed{s}/imputasi_metrics.json'
        if os.path.exists(done) and not a.ulang:
            print(f'[{k}/{len(jobs)}] lewati diffputer | {d} | seed{s} (sudah ada; pakai --ulang untuk mengulang)')
            dilewati.append((d, s))
            continue
        print(f'\n[{k}/{len(jobs)}] mulai diffputer | {d} | seed{s}', flush=True)
        try:
            run_one(d, s, a)
            selesai.append((d, s))
        except Exception as e:                                            # satu run gagal tidak menghentikan yang lain
            import traceback
            traceback.print_exc()
            print(f'[GAGAL] diffputer | {d} | seed{s}: {e}', flush=True)
            gagal.append((d, s))
    print(f'\nRINGKASAN: selesai {len(selesai)}, dilewati {len(dilewati)}, gagal {len(gagal)}')
    for d, s in gagal:
        print(f'  gagal: --dataset {d} --seed {s}')


if __name__ == '__main__':
    main()