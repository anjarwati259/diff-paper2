"""
BASELINE DUA TAHAP — TAHAP 1: IMPUTASI FITUR (cara klasik, tanpa kebocoran val/test).
Tahap 2 (prediksi SVM) ada di run_svm.py dan membaca CSV hasil skrip ini.

  --imputer     kode                                                   acuan
  mean          rata-rata (numerik) / modus (kategorik) dari TRAIN
  knn           sklearn.impute.KNNImputer (k = 5, bobot seragam, jarak       Troyanskaya dkk., Bioinformatics 2001
                nan_euclidean), di-fit di TRAIN (tetangga = baris train); kategorik masuk sebagai one-hot
                (kode kategori tidak dianggap berurutan), isian kategorik = MODUS tetangga
  mice         hyperimpute plugin 'mice' (setting bawaan)             van Buuren & Groothuis-Oudshoorn 2011
  missforest    hyperimpute plugin 'sklearn_missforest' (bawaan)       Stekhoven & Bühlmann 2012
  hyperimpute   hyperimpute plugin 'hyperimpute' (bawaan)              Jarrett dkk., ICML 2022
  miwae         hyperimpute plugin 'miwae' (bawaan)                    Mattei & Frellsen, ICML 2019
  (nama plugin & setting bawaan sama dengan baselines/hyperimpute/hyperimpute_benchmark.py di repo DiffPuter)

ATURAN ANTI-KEBOCORAN
  1. Imputer di-FIT HANYA di fitur TRAIN. Kolom target (y) tidak pernah masuk ke imputer.
  2. Val dan test hanya di-TRANSFORM dengan imputer yang sudah di-fit. Setiap baris val/test diisi dari nilai
     teramati baris itu sendiri + model dari train; baris val/test lain tidak dipakai.
       - mean, mice, missforest, miwae : fit(train) lalu transform(val), transform(test) dari paket.
       - knn : KNNImputer.fit(train) lalu transform(train/val/test) -> tetangga (donor) HANYA baris train.
       - hyperimpute : paket hanya punya fit_transform (selalu melatih ulang). Untuk val/test dipakai model
         per kolom yang sudah dilatih di train, diterapkan berputar (chained equations) sampai stabil — sama
         seperti transform() pada IterativeImputer. Tidak ada pelatihan ulang di val/test.
  3. Standardisasi numerik & daftar kategori dihitung dari train.

REPRESENTASI UNTUK IMPUTER
  numerik  : (x - mean_train) / std_train          kategorik : kode bilangan bulat 0..k-1 (kategori dari train)
  Setelah imputasi dikembalikan ke NILAI ASLI: numerik dikali balik lalu dibulatkan ke JUMLAH DESIMAL kolom
  aslinya (maksimum di nilai teramati train; kolom bilangan bulat -> 0 desimal),
  kategorik dibulatkan ke kode valid terdekat lalu diubah ke label aslinya. Sel yang teramati TIDAK diubah.

KELUARAN  (results/imputasi/{imputer}/{dataset}/MAR{rr}/seed{k}/)
  missing_{train,val,test}.csv  : data asli dengan sel yang di-mask dikosongkan (sebelum imputasi)
  imputed_{train,val,test}.csv  : data setelah imputasi, NILAI ASLI, + kolom target asli (tidak diubah)
  imputasi_metrics.json         : mutu imputasi per bagian — MAE/RMSE numerik (satuan std train, sama dengan
                                  metrik imputasi metode usulan) & ACC kategorik — serta waktu
  result.txt                    : ringkasan (mode append)

Pemakaian (dari folder baselines_eksternal):
  python run_imputasi.py                       # SEMUA: DAFTAR_IMPUTER × DAFTAR_DATASET × DAFTAR_SEED (atur di kode)
  python run_imputasi.py --imputer mean mice   # sebagian imputer saja (dataset & seed tetap dari daftar)
  python run_imputasi.py --imputer missforest --dataset bike --seed 0      # satu run
  Run yang hasilnya sudah ada otomatis dilewati (pakai --ulang untuk mengulang). Run yang gagal tidak
  menghentikan run lain; daftarnya dicetak di akhir.
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
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from data_loader import load_split                                           # noqa: E402

IMPUTERS = ['mean', 'knn', 'mice', 'missforest', 'hyperimpute', 'miwae']
PLUGIN = {'mice': 'mice', 'missforest': 'sklearn_missforest', 'hyperimpute': 'hyperimpute', 'miwae': 'miwae'}
PARTS = ['train', 'val', 'test']


def _patch_hyperimpute_compat():
    """Kompatibilitas hyperimpute dengan pustaka versi baru TANPA mengubah algoritmanya:
    (1) scikit-learn >= 1.8 menghapus argumen LogisticRegression(multi_class=...) yang masih dikirim hyperimpute
        -> argumen itu diabaikan (perilakunya sama dengan bawaan baru: multinomial).
    (2) pandas >= 3 (copy-on-write): penulisan berantai X[col][mask] = ... di _impute_single_column TIDAK lagi
        mengubah X, sehingga HyperImpute diam-diam tidak mengisi apa pun -> ditulis ulang dengan X.loc[mask, col].
    Pada versi lama kedua tambalan tidak mengubah hasil."""
    import functools
    import inspect
    import sklearn.linear_model as lm
    LR = lm.LogisticRegression
    if 'multi_class' not in inspect.signature(LR.__init__).parameters and not getattr(LR.__init__, '_tambal', False):
        orig = LR.__init__

        @functools.wraps(orig)
        def init(self, *args, **kw):
            kw.pop('multi_class', None)
            orig(self, *args, **kw)
        init._tambal = True
        LR.__init__ = init

    from hyperimpute.plugins.imputers import _hyperimpute_internals as hi
    if getattr(hi.IterativeErrorCorrection, '_tambal', False):
        return

    def _impute_single_column(self, X, col, train):
        if self.mask[col].sum() == 0:
            return X
        cov_cols = self._get_neighbors_for_col(col)
        covs = X[cov_cols]
        target = X[col]
        m = self.mask[col].to_numpy()
        X_train, y_train = covs[~m], target[~m]
        if col in self.categorical_cols:
            y_train = y_train.astype(int)
        if len(np.unique(y_train)) == 1:
            X.loc[m, col] = np.asarray(y_train)[0]
            return X
        est = self.column_to_model[col]
        if train:
            est.fit(X_train, y_train)
        col_min, col_max = self.limits[col]
        X.loc[m, col] = np.clip(np.asarray(est.predict(covs[m])).squeeze(), col_min, col_max)
        return X
    hi.IterativeErrorCorrection._impute_single_column = _impute_single_column
    hi.IterativeErrorCorrection._tambal = True


def get_imputers():
    try:
        from hyperimpute.plugins.imputers import Imputers
    except ImportError:
        src = os.path.join(PROJECT_DIR, 'baselines_eksternal', 'hyperimpute', 'src')
        if not os.path.isdir(src):
            raise ImportError('Paket hyperimpute tidak terpasang: pip install hyperimpute')
        sys.path.insert(0, src)
        from hyperimpute.plugins.imputers import Imputers
    _patch_hyperimpute_compat()
    return Imputers()


# =============================================================================================
# Kodekan tabel asli -> matriks untuk imputer, dan sebaliknya (semua statistik dari TRAIN)
# =============================================================================================
def n_decimals(series, maks=12):
    """Banyak angka di belakang koma yang dipakai kolom ini (maksimum atas nilai unik yang teramati)."""
    v = pd.Series(series).dropna().unique()
    if len(v) == 0 or pd.api.types.is_integer_dtype(pd.Series(series).dtype):
        return 0
    d = 0
    for x in v:
        t = np.format_float_positional(float(x), trim='-')               # tanpa notasi e, tanpa nol di ujung
        if '.' in t:
            d = max(d, len(t.split('.')[1]))
            if d >= maks:
                return maks
    return d


class Encoder:
    def __init__(self, df_train, num_cols, cat_cols, dtypes):
        """df_train: train dengan sel hilang = NaN; dtypes: tipe kolom ASLI (sebelum masking, karena masking
        mengubah kolom bilangan bulat menjadi float)."""
        self.num_cols, self.cat_cols = list(num_cols), list(cat_cols)
        self.dtypes = dict(dtypes)
        self.mean = {c: float(df_train[c].mean()) for c in self.num_cols}
        self.std = {c: float(df_train[c].std()) or 1.0 for c in self.num_cols}
        # jumlah angka di belakang koma per kolom numerik = maksimum di nilai teramati TRAIN
        # (mis. kolom asli 2 desimal -> hasil imputasi dibulatkan 2 desimal; kolom bilangan bulat -> 0)
        self.decimals = {c: n_decimals(df_train[c]) for c in self.num_cols}
        self.cats = {}
        for c in self.cat_cols:
            v = df_train[c].dropna().unique().tolist()
            if pd.api.types.is_integer_dtype(self.dtypes[c]):
                v = [int(x) for x in v]
            self.cats[c] = sorted(set(v), key=lambda x: (str(type(x)), x))

    def encode(self, df):
        """-> DataFrame numerik; NaN = hilang. Kategori yang tidak ada di train ikut NaN (hanya untuk imputer)."""
        out = pd.DataFrame(index=range(len(df)))
        for c in self.num_cols:
            out[c] = (df[c].to_numpy(np.float64) - self.mean[c]) / self.std[c]
        for c in self.cat_cols:
            idx = {v: i for i, v in enumerate(self.cats[c])}
            isint = pd.api.types.is_integer_dtype(self.dtypes[c])
            out[c] = [idx.get(int(v) if isint else v, np.nan) if pd.notna(v) else np.nan for v in df[c].tolist()]
            out[c] = out[c].astype(np.float64)
        return out

    def decode(self, Z):
        """Matriks hasil imputer -> DataFrame NILAI ASLI."""
        out = pd.DataFrame(index=range(len(Z)))
        for c in self.num_cols:
            v = np.round(Z[c].to_numpy(np.float64) * self.std[c] + self.mean[c], self.decimals[c])
            out[c] = v.astype(self.dtypes[c]) if pd.api.types.is_integer_dtype(self.dtypes[c]) else v
        for c in self.cat_cols:
            k = np.clip(np.round(Z[c].to_numpy(np.float64)), 0, len(self.cats[c]) - 1).astype(int)
            out[c] = pd.Series(np.array(self.cats[c], dtype=object)[k]).astype(self.dtypes[c])
        return out


# =============================================================================================
# Imputer: fit di train, transform di val/test
# =============================================================================================
class MeanMode:
    def fit(self, Z, cat_cols):
        self.fill = {c: (Z[c].mode(dropna=True).iloc[0] if c in cat_cols else Z[c].mean()) for c in Z.columns}
        return self

    def transform(self, Z):
        return Z.fillna(self.fill)


class KNNMixed:
    """KNN imputer untuk data campuran memakai PACKAGE sklearn.impute.KNNImputer
    (n_neighbors=k, weights='uniform', metric='nan_euclidean'), di-fit HANYA di TRAIN (donor = baris train).
    Input = matriks terkode Encoder: numerik (x - mean_train) / std_train, kategorik kode 0..k-1 (NaN = hilang).
    Penanganan kategorik (agar kode kategori tidak dianggap berurutan):
      - sebelum masuk KNNImputer, tiap kolom kategorik diubah ke one-hot dibagi sqrt(2) (beda satu kategori =
        jarak 1, setara 1 std numerik); sel hilang = NaN di seluruh blok one-hot-nya;
      - KNNImputer mengisi blok one-hot dengan rata-rata donor = proporsi tiap kategori di k tetangga, lalu
        diambil kategori dengan proporsi terbesar (argmax) = MODUS tetangga (seri -> kode terkecil).
    Sel yang teramati tidak diubah."""
    def __init__(self, k=5):
        self.k = k

    def _feat(self, Z):
        parts = [Z[self.num].to_numpy(np.float64)] if self.num else []
        for c in self.cat:
            v = Z[c].to_numpy(np.float64)
            oh = (v[:, None] == self.levels[c][None, :]).astype(np.float64) / np.sqrt(2.0)
            oh[np.isnan(v)] = np.nan
            parts.append(oh)
        return np.concatenate(parts, 1)

    def fit(self, Z, cat_cols):
        from sklearn.impute import KNNImputer
        self.cols = list(Z.columns)
        self.cat = [c for c in self.cols if c in cat_cols]
        self.num = [c for c in self.cols if c not in cat_cols]
        self.levels = {c: np.unique(Z[c].dropna().to_numpy(np.float64)) for c in self.cat}
        self.imp = KNNImputer(n_neighbors=self.k, weights='uniform', metric='nan_euclidean',
                              keep_empty_features=True)
        self.imp.fit(self._feat(Z.reset_index(drop=True)))
        return self

    def transform(self, Z):
        Z = Z.reset_index(drop=True)
        F = self.imp.transform(self._feat(Z))
        out = Z.copy().astype(np.float64)
        n_num = len(self.num)
        if self.num:
            hil = Z[self.num].isna().to_numpy()
            V = out[self.num].to_numpy(copy=True)
            V[hil] = F[:, :n_num][hil]                                        # sel teramati tidak diubah
            out[self.num] = V
        pos = n_num
        for c in self.cat:
            L = len(self.levels[c])
            code = self.levels[c][F[:, pos:pos + L].argmax(1)]
            hil = Z[c].isna().to_numpy()
            col = out[c].to_numpy(copy=True)
            col[hil] = code[hil]
            out[c] = col
            pos += L
        return out[self.cols]


def hyperimpute_transform(plugin, Z):
    """Terapkan model per kolom HyperImpute yang sudah dilatih di train ke data baru, tanpa pelatihan ulang.
    Mengikuti _setup / _impute_single_column / _tear_down di _hyperimpute_internals.py, dengan statistik train."""
    m = plugin.model
    X = Z.copy().reset_index(drop=True)
    X.columns = X.columns.map(str)
    mask = X.isna()
    for col, le in m.encoders.items():                                   # kode kategori train
        ok = X[col].notna() & X[col].isin(le.classes_)
        X.loc[~ok, col] = np.nan
        mask[col] = mask[col] | ~ok
        X.loc[ok, col] = le.transform(X.loc[ok, col])
    Xt = pd.DataFrame(np.asarray(m.baseline_imputer.transform(X)), columns=X.columns)   # isian awal dari train
    for _ in range(m.n_inner_iter):
        prev = Xt.copy()
        for col in m.imputation_order:
            rows = mask[col].to_numpy()
            if not rows.any() or col not in m.column_to_model:
                continue                                                  # kolom tanpa model: isian awal (train)
            covs = Xt[[c for c in m.all_cols if c != col]]
            pred = np.asarray(m.column_to_model[col].predict(covs[rows])).squeeze()
            lo, hi = m.limits[col]
            Xt.loc[rows, col] = np.clip(pred, lo, hi)
        if np.linalg.norm((Xt - prev).to_numpy(), ord=np.inf) < 1e-8:
            break
    for col, le in m.encoders.items():
        Xt[col] = le.inverse_transform(Xt[col].astype(int))
    return Xt


def impute(name, Z, cat_cols, seed):
    """Z: dict part -> DataFrame terkode (NaN = hilang). Fit HANYA di Z['train']."""
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    cols = list(Z['train'].columns)
    if name == 'mean':
        imp = MeanMode().fit(Z['train'], cat_cols)
        return {p: imp.transform(Z[p]) for p in PARTS}
    if name == 'knn':
        imp = KNNMixed(k=5).fit(Z['train'], cat_cols)
        return {p: imp.transform(Z[p]) for p in PARTS}
    plugin = get_imputers().get(PLUGIN[name], random_state=seed)
    Ztr = Z['train'].copy(); Ztr.columns = Ztr.columns.map(str)
    if name == 'hyperimpute':
        out = {'train': plugin.fit_transform(Ztr)}                        # melatih di train
        out.update({p: hyperimpute_transform(plugin, Z[p]) for p in ('val', 'test')})
    else:
        plugin.fit(Ztr)
        out = {p: plugin.transform(Z[p].set_axis(Ztr.columns, axis=1)) for p in PARTS}
    res = {}
    for p in PARTS:
        o = pd.DataFrame(np.asarray(out[p], dtype=np.float64), columns=cols)
        res[p] = Z[p].where(Z[p].notna(), o)                             # sel teramati tetap nilai asli
    return res


# =============================================================================================
# Mutu imputasi (nilai asli; numerik dalam satuan std train = skala metrik imputasi metode usulan)
# =============================================================================================
def imputation_quality(df_imp, df_true, mask_feat, enc):
    num_err, hits = [], []
    for j, c in enumerate(enc.num_cols):
        m = mask_feat[:, j]
        if m.any():
            num_err.append((df_imp[c].to_numpy(np.float64)[m] - df_true[c].to_numpy(np.float64)[m]) / enc.std[c])
    for j, c in enumerate(enc.cat_cols):
        m = mask_feat[:, len(enc.num_cols) + j]
        if m.any():
            hits.append(df_imp[c].astype(str).to_numpy()[m] == df_true[c].astype(str).to_numpy()[m])
    e = np.concatenate(num_err) if num_err else np.array([np.nan])
    return {'mae': float(np.abs(e).mean()), 'rmse': float(np.sqrt((e ** 2).mean())),
            'acc': float(np.concatenate(hits).mean()) if hits else float('nan'),
            'n_sel_numerik': int(sum(len(x) for x in num_err)), 'n_sel_kategorik': int(sum(len(x) for x in hits))}


# =============================================================================================
# DAFTAR YANG DIJALANKAN — ubah di sini (atau timpa lewat argumen --imputer / --dataset / --seed)
# =============================================================================================
DAFTAR_IMPUTER = ['mean', 'knn', 'mice', 'missforest', 'hyperimpute', 'miwae']
DAFTAR_DATASET = ['bike', 'shoppers']
DAFTAR_SEED = [0, 1, 2, 3, 4]
RASIO = 0.3


def run_one(imputer, dataset, seed, ratio, mech, out_root):
    t_all = time.time()
    data, meta = load_split(dataset, seed, ratio, mech)
    num_cols, cat_cols, target = list(meta['num_cols']), list(meta['cat_cols']), meta['target']
    feat = num_cols + cat_cols
    tag = f'{mech}{int(round(ratio * 100)):02d}'
    out_dir = f'{out_root}/imputasi/{imputer}/{dataset}/{tag}/seed{seed}'
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== imputasi {imputer} | {dataset} ({meta["task"]}) | {tag} seed{seed} | '
          f'missing train {data["train"]["mask_feat"].mean():.3f} ===', flush=True)

    # tabel asli dengan sel yang di-mask dikosongkan
    miss = {p: data[p]['df'][feat].mask(np.asarray(data[p]['mask_feat'], dtype=bool)).reset_index(drop=True)
            for p in PARTS}
    enc = Encoder(miss['train'], num_cols, cat_cols, data['train']['df'][feat].dtypes.to_dict())   # statistik TRAIN
    Z = {p: enc.encode(miss[p]) for p in PARTS}

    t0 = time.time()
    Zi = impute(imputer, Z, cat_cols, seed)
    t_imp = time.time() - t0

    summary = {'imputer': imputer, 'plugin_hyperimpute': PLUGIN.get(imputer), 'dataset': dataset,
               'task': meta['task'], 'mech': mech, 'ratio': ratio, 'seed': seed,
               'aturan': 'fit di train saja; val/test hanya transform; y tidak dipakai',
               'imputasi_time_s': t_imp, 'kolom_numerik': num_cols, 'kolom_kategorik': cat_cols}
    for p in PARTS:
        df_imp = enc.decode(Zi[p])
        df_true = data[p]['df'][feat].reset_index(drop=True)
        obs = ~np.asarray(data[p]['mask_feat'], dtype=bool)
        for j, c in enumerate(feat):                                      # sel teramati = nilai asli persis
            df_imp.loc[obs[:, j], c] = df_true.loc[obs[:, j], c]
        assert not df_imp.isna().any().any(), f'masih ada sel kosong di {p}'
        df_imp[target] = data[p]['df'][target].to_numpy()
        m_out = miss[p].copy(); m_out[target] = data[p]['df'][target].to_numpy()
        m_out.to_csv(f'{out_dir}/missing_{p}.csv', index=False)
        df_imp.to_csv(f'{out_dir}/imputed_{p}.csv', index=False)
        q = imputation_quality(df_imp, df_true, np.asarray(data[p]['mask_feat'], dtype=bool), enc)
        summary[p] = q
        summary['desimal_per_kolom'] = enc.decimals
        print(f'Imputasi ({p}) → MAE={q["mae"]:.4f}  RMSE={q["rmse"]:.4f}  ACC={q["acc"]:.4f}', flush=True)
    summary['total_time_s'] = time.time() - t_all
    with open(f'{out_dir}/imputasi_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    te = summary['test']
    with open(f'{out_dir}/result.txt', 'a') as f:
        f.write(f"test | imputasi {imputer} | MAE {te['mae']:.4f} RMSE {te['rmse']:.4f} ACC {te['acc']:.4f} | "
                f"waktu {summary['total_time_s']:.0f}s\n")
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {out_dir}/', flush=True)


def main():
    ap = argparse.ArgumentParser(description='Tahap 1 baseline dua tahap: imputasi fitur (fit di train)')
    ap.add_argument('--imputer', nargs='+', default=DAFTAR_IMPUTER, choices=IMPUTERS,
                    help='bawaan: DAFTAR_IMPUTER di kode (semua)')
    ap.add_argument('--dataset', nargs='+', default=DAFTAR_DATASET, help='bawaan: DAFTAR_DATASET di kode')
    ap.add_argument('--seed', nargs='+', type=int, default=DAFTAR_SEED, help='bawaan: DAFTAR_SEED di kode')
    ap.add_argument('--ratio', type=float, default=RASIO)
    ap.add_argument('--mech', default='MAR')
    ap.add_argument('--ulang', action='store_true', help='jalankan ulang walau hasilnya sudah ada')
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'results'),
                    help='folder hasil (bawaan: results/ di folder tempat skrip ini berada)')
    a = ap.parse_args()

    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    jobs = [(m, d, s) for m in a.imputer for d in a.dataset for s in a.seed]
    print(f'Total {len(jobs)} run: imputer {a.imputer} × dataset {a.dataset} × seed {a.seed} | {tag}', flush=True)
    selesai, dilewati, gagal = [], [], []
    for k, (m, d, s) in enumerate(jobs, 1):
        done = f'{a.out_dir}/imputasi/{m}/{d}/{tag}/seed{s}/imputasi_metrics.json'
        if os.path.exists(done) and not a.ulang:
            print(f'[{k}/{len(jobs)}] lewati {m} | {d} | seed{s} (sudah ada; pakai --ulang untuk mengulang)', flush=True)
            dilewati.append((m, d, s))
            continue
        print(f'\n[{k}/{len(jobs)}] mulai {m} | {d} | seed{s}', flush=True)
        try:
            run_one(m, d, s, a.ratio, a.mech, a.out_dir)
            selesai.append((m, d, s))
        except Exception as e:                                            # satu run gagal tidak menghentikan yang lain
            import traceback
            traceback.print_exc()
            print(f'[GAGAL] {m} | {d} | seed{s}: {e}', flush=True)
            gagal.append((m, d, s))
    print(f'\nRINGKASAN: selesai {len(selesai)}, dilewati {len(dilewati)}, gagal {len(gagal)}')
    for m, d, s in gagal:
        print(f'  gagal: --imputer {m} --dataset {d} --seed {s}')


if __name__ == '__main__':
    main()