"""
Pemuat data untuk Paper 2 (DiffPuter untuk prediksi target regresi).

Membaca hasil download_and_split.py:
  datasets/Info/{nama}.json
  datasets/{nama}/splits/seed{k}/{train,val,test}.csv
  datasets/{nama}/masks/MAR{rr}/seed{k}/{train,val,test}_mask.npy

Lalu menyiapkan matriks untuk diffusion, mengikuti dataset.py + main.py DiffPuter:
  - Kolom: [numerik | kategorik (binary encoding / analog bits) | y]
  - Mask kolom kategorik diperluas ke semua bit-nya (extend_mask DiffPuter)
  - Standardisasi pakai mean/std nilai TERAMATI di train, lalu dibagi 2 (main.py baris 58)
  - y di-log (log1p) dulu bila log_target = true, lalu distandardisasi

Tambahan dibanding DiffPuter:
  - Kolom y ikut di matriks (kolom terakhir).
  - Mask y: train = False (y diketahui), val/test = True (y ditebak).
  - Decode kategorik memilih kategori valid terdekat (kode biner yang tidak valid tidak dipakai).

Target KLASIFIKASI (Info JSON "task_type": "classification"):
  - y = blok one-hot K kolom di akhir matriks (CARD, Han dkk. NeurIPS 2022), nilai +0.5 (kelas benar)
    dan -0.5 (kelas lain). Blok ini TIDAK distandardisasi (mean 0, std 0.5 ditetapkan, sehingga
    nilai di matriks tetap +-0.5, selaras sigma_data = 0.5 EDM). Kelas langka tidak membesar.
  - meta['y_idx'] = kolom PERTAMA blok y; meta['y_cols'] = semua kolom y (regresi: [y_idx]).
  - meta['classes'] = label kelas (urut naik); inverse_transform -> label kelas lewat argmax.
  Jalur regresi tidak berubah (hasil identik dengan versi sebelumnya).

Transformasi fitur numerik (num_transform, default 'none' = identik dengan versi sebelumnya):
  'quantile'   : QuantileTransformer -> distribusi normal (TabDDPM, Kotelnikov dkk. ICML 2023)
  'yeojohnson' : PowerTransformer Yeo-Johnson (pembanding)
  Di-fit HANYA pada nilai teramati di train, diterapkan ke SEMUA fitur numerik (bukan pilihan per kolom),
  lalu distandardisasi (/2) seperti biasa. Kategorik & y tidak disentuh.
  Dua skala:
    X_obs (input model) & batas clip  -> skala MODEL (hasil transformasi)
    X_true, meta['mean'/'std']        -> skala ASLI (z-score nilai asli, sama seperti 'none')
  Hasil model dikembalikan ke skala asli dengan model_to_eval() sebelum metrik imputasi / CSV, sehingga
  MAE/RMSE imputasi tetap sebanding dengan DiffPuter asli dan run lama.

Isi yang dikembalikan untuk tiap bagian (train / val / test):
  X_true : matriks lengkap terstandardisasi (HANYA untuk evaluasi, jangan diberikan ke model)
  X_obs  : sama dengan X_true, tetapi sel yang hilang/ditebak diisi 0 (= mean). Ini input model.
  mask   : True = hilang/ditebak (fitur yang di-mask + kolom y untuk val/test)
  mask_feat : mask fitur saja (tanpa kolom y)
  df     : DataFrame asli bagian itu (skala asli, untuk CSV hasil dan evaluasi)

Pemakaian di kode lain:
  from data_loader import load_split
  data, meta = load_split('bike', seed=0, ratio=0.3)
  X_train = data['train']['X_obs']; mask_train = data['train']['mask']

Cek cepat (jalankan sebelum lanjut ke langkah berikutnya):
  python data_loader.py --dataset bike --seed 0 --ratio 0.3
  python data_loader.py --all          # semua dataset x seed x ratio
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

CLIP_MARGIN = 0.05   # margin batas clip sampel = 5% rentang nilai train (per kolom)

DATA_DIR = 'datasets'
PARTS = ['train', 'val', 'test']


# --------------------------------------------------------------------------
# Binary encoding (sama seperti DiffPuter)
# --------------------------------------------------------------------------
def n_bits(n_categories):
    return max(1, (n_categories - 1).bit_length())


def build_cat_maps(df_all, cat_cols):
    """Daftar kategori per kolom dari seluruh data.csv, urut sesuai kemunculan pertama
    (sama dengan dataset.py DiffPuter: data_cat[column].unique())."""
    maps = {}
    for c in cat_cols:
        cats = df_all[c].dropna().unique().tolist()
        bits = n_bits(len(cats))
        codes = np.array([[int(b) for b in format(i, f'0{bits}b')] for i in range(len(cats))],
                         dtype=np.float64)
        maps[c] = {'categories': cats, 'bits': bits, 'codes': codes}
    return maps


def encode_cats(df, cat_cols, cat_maps):
    blocks = []
    for c in cat_cols:
        m = cat_maps[c]
        idx = {v: i for i, v in enumerate(m['categories'])}
        block = np.zeros((len(df), m['bits']))
        for r, v in enumerate(df[c].tolist()):
            if pd.isna(v):
                block[r] = np.nan
            else:
                block[r] = m['codes'][idx[v]]
        blocks.append(block)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0))


def decode_cats(bits_matrix, cat_cols, cat_maps):
    """Bit (skala 0..1) -> kategori. Pilih kategori valid dengan kode biner terdekat."""
    out, start = {}, 0
    for c in cat_cols:
        m = cat_maps[c]
        b = bits_matrix[:, start:start + m['bits']]
        dist = ((b[:, None, :] - m['codes'][None, :, :]) ** 2).sum(axis=2)
        out[c] = np.array(m['categories'], dtype=object)[dist.argmin(axis=1)]
        start += m['bits']
    return out


def extend_mask(mask_feat, n_num, cat_bits):
    """Mask per kolom asli -> mask per kolom matriks (kategorik diperluas ke tiap bit)."""
    parts = [mask_feat[:, :n_num]]
    for j, bits in enumerate(cat_bits):
        parts.append(np.repeat(mask_feat[:, n_num + j:n_num + j + 1], bits, axis=1))
    return np.concatenate(parts, axis=1)


# --------------------------------------------------------------------------
# Pemuat utama
# --------------------------------------------------------------------------
def make_num_transformer(kind, seed=0):
    if kind == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution='normal', n_quantiles=1000, subsample=10 ** 9,
                                   random_state=seed)
    if kind == 'yeojohnson':
        from sklearn.preprocessing import PowerTransformer
        return PowerTransformer(method='yeo-johnson', standardize=False)
    raise ValueError(f'num_transform tidak dikenal: {kind}')


def load_split(name, seed, ratio, mech='MAR', data_dir=DATA_DIR, num_transform='none'):
    with open(f'{data_dir}/Info/{name}.json') as f:
        info = json.load(f)
    num_cols, cat_cols, target = info['num_cols'], info['cat_cols'], info['target']
    task = info.get('task_type', 'regression')
    if task not in ('regression', 'classification'):
        raise ValueError(f"task_type tidak dikenal: {task}")
    is_cls = task == 'classification'
    log_target = bool(info.get('log_target', False)) and not is_cls
    n_num, n_feat = len(num_cols), len(num_cols) + len(cat_cols)

    df_all = pd.read_csv(f'{data_dir}/{name}/data.csv')
    cat_maps = build_cat_maps(df_all, cat_cols)
    cat_bits = [cat_maps[c]['bits'] for c in cat_cols]
    classes = sorted(df_all[target].dropna().unique().tolist()) if is_cls else None
    n_y = len(classes) if is_cls else 1

    tag = f'{mech}{int(round(ratio * 100)):02d}'
    raw = {}
    for part in PARTS:
        df = pd.read_csv(f'{data_dir}/{name}/splits/seed{seed}/{part}.csv')

        if ratio > 0:
            mask_feat = np.load(f'{data_dir}/{name}/masks/{tag}/seed{seed}/{part}_mask.npy')
        else:
            mask_feat = np.zeros((len(df), n_feat), dtype=bool)
        mask_feat = mask_feat | df[num_cols + cat_cols].isna().to_numpy()   # missing asli ikut ditandai

        if is_cls:
            Y = encode_onehot(df[target], classes)                          # [n, K], +-0.5
        else:
            y = df[target].to_numpy(dtype=np.float64)
            if log_target:
                y = np.log1p(y)
            Y = y[:, None]

        X = np.concatenate([df[num_cols].to_numpy(dtype=np.float64),
                            encode_cats(df, cat_cols, cat_maps),
                            Y], axis=1)

        mask = np.concatenate([extend_mask(mask_feat, n_num, cat_bits),
                               np.full((len(df), n_y), part != 'train')], axis=1)
        raw[part] = (df, X, mask, mask_feat)

    # mean/std dari nilai TERAMATI di train (mean_std DiffPuter); y train selalu teramati
    _, X_tr, m_tr, _ = raw['train']
    obs = (~m_tr) & ~np.isnan(X_tr)
    cnt = np.maximum(obs.sum(0), 1)
    mean = np.where(obs, X_tr, 0).sum(0) / cnt
    std = np.sqrt((np.where(obs, X_tr - mean, 0) ** 2).sum(0) / cnt)
    std[std == 0] = 1.0
    y_idx = X_tr.shape[1] - n_y
    if is_cls:                         # blok one-hot tidak distandardisasi: (x - 0) / 0.5 / 2 = x
        mean[y_idx:] = 0.0
        std[y_idx:] = 0.5

    # batas clip sampel (skala terstandardisasi): rentang nilai TERAMATI di train +- 5% rentang
    Z = np.where(obs, (X_tr - mean) / std / 2.0, np.nan)
    zmin, zmax = np.nanmin(Z, axis=0), np.nanmax(Z, axis=0)
    margin = CLIP_MARGIN * (zmax - zmin)
    clip_lo, clip_hi = zmin - margin, zmax + margin

    # ---- transformasi fitur numerik (skala MODEL); 'none' -> skala model = skala asli ----
    tf = None
    if num_transform != 'none' and n_num > 0:
        Xn_tr = np.where(obs[:, :n_num], X_tr[:, :n_num], np.nan)          # hanya nilai teramati train
        tfm = make_num_transformer(num_transform, seed)
        n_obs_min = int(obs[:, :n_num].sum(0).min())
        if num_transform == 'quantile':
            tfm.set_params(n_quantiles=max(10, min(1000, n_obs_min)))
        tfm.fit(Xn_tr)
        U_tr = tfm.transform(Xn_tr)
        mean_u = np.nanmean(U_tr, axis=0)
        std_u = np.nanstd(U_tr, axis=0)
        std_u[~(std_u > 0)] = 1.0
        tf = {'kind': num_transform, 'tfm': tfm, 'mean_u': mean_u, 'std_u': std_u}

        def to_model(X):
            Z = (X - mean) / std / 2.0
            U = tfm.transform(X[:, :n_num])
            Z[:, :n_num] = (U - mean_u) / std_u / 2.0
            return Z
        Zm = np.where(obs, to_model(X_tr), np.nan)                       # NaN dipertahankan transformer
        zmin, zmax = np.nanmin(Zm, axis=0), np.nanmax(Zm, axis=0)       # batas clip di skala MODEL
        margin = CLIP_MARGIN * (zmax - zmin)
        clip_lo, clip_hi = zmin - margin, zmax + margin

    data = {}
    for part in PARTS:
        df, X, mask, mask_feat = raw[part]
        X_true = (X - mean) / std / 2.0                                    # skala ASLI (evaluasi)
        X_model = X_true if tf is None else to_model(X)                     # skala MODEL (input)
        X_obs = np.where(mask | np.isnan(X_model), 0.0, X_model)
        data[part] = {'df': df, 'X_true': X_true, 'X_obs': X_obs,
                      'mask': mask, 'mask_feat': mask_feat}

    meta = {
        'name': name, 'seed': seed, 'ratio': ratio, 'mech': mech,
        'num_cols': num_cols, 'cat_cols': cat_cols, 'target': target,
        'log_target': log_target, 'n_num': n_num, 'cat_bits': cat_bits,
        'cat_maps': cat_maps, 'y_idx': y_idx, 'dim': X_tr.shape[1],
        'mean': mean, 'std': std, 'clip_lo': clip_lo, 'clip_hi': clip_hi,
        'task': task, 'y_cols': list(range(y_idx, X_tr.shape[1])), 'classes': classes,
        'num_transform': num_transform, 'tf': tf,
    }
    return data, meta


def model_to_eval(Xs, meta):
    """Matriks skala MODEL (hasil model, mis. rata-rata sampel / isian E-step) -> skala ASLI (z-score nilai asli),
    agar metrik imputasi & CSV sebanding dengan 'none'. Tanpa transformasi: dikembalikan apa adanya (objek sama).
    Hanya kolom fitur numerik yang berubah."""
    tf = meta.get('tf')
    if tf is None:
        return Xs
    n = meta['n_num']
    out = np.array(Xs, dtype=np.float64, copy=True)
    U = out[..., :n] * 2.0 * tf['std_u'] + tf['mean_u']
    shp = U.shape
    Xraw = tf['tfm'].inverse_transform(U.reshape(-1, n)).reshape(shp)
    out[..., :n] = (Xraw - meta['mean'][:n]) / meta['std'][:n] / 2.0
    return out


def encode_onehot(series, classes):
    """Label kelas -> one-hot +-0.5 [n, K]. Label di luar daftar kelas -> error."""
    idx = {v: i for i, v in enumerate(classes)}
    lab = series.tolist()
    miss = [v for v in lab if v not in idx]
    if miss:
        raise ValueError(f'label target tidak dikenal: {miss[:5]}')
    Y = np.full((len(lab), len(classes)), -0.5)
    Y[np.arange(len(lab)), [idx[v] for v in lab]] = 0.5
    return Y


def y_true_index(df, meta):
    """Indeks kelas benar (0..K-1) untuk DataFrame bagian mana pun (klasifikasi)."""
    idx = {v: i for i, v in enumerate(meta['classes'])}
    return np.array([idx[v] for v in df[meta['target']].tolist()], dtype=np.int64)


def clip_bounds(meta):
    return meta['clip_lo'], meta['clip_hi']


def inverse_transform(Xs, meta):
    """Matriks terstandardisasi -> DataFrame skala asli (numerik, kategorik, y)."""
    X = Xs * 2.0 * meta['std'] + meta['mean']
    n_num, y_idx = meta['n_num'], meta['y_idx']
    out = pd.DataFrame(X[:, :n_num], columns=meta['num_cols'])
    cats = decode_cats(X[:, n_num:y_idx], meta['cat_cols'], meta['cat_maps'])
    for c in meta['cat_cols']:
        out[c] = cats[c]
    if meta.get('task', 'regression') == 'classification':
        out[meta['target']] = np.array(meta['classes'], dtype=object)[X[:, y_idx:].argmax(axis=1)]
        return out
    y = X[:, y_idx]
    out[meta['target']] = np.expm1(y) if meta['log_target'] else y
    return out


def inverse_y(y_std, meta):
    """Kolom y terstandardisasi (array apa pun bentuknya) -> skala asli. Hanya regresi."""
    if meta.get('task', 'regression') == 'classification':
        raise ValueError('inverse_y hanya untuk regresi; klasifikasi memakai blok one-hot meta["y_cols"]')
    i = meta['y_idx']
    y = y_std * 2.0 * meta['std'][i] + meta['mean'][i]
    return np.expm1(y) if meta['log_target'] else y


# --------------------------------------------------------------------------
# Cek
# --------------------------------------------------------------------------
def check(name, seed, ratio, verbose=True, num_transform='none'):
    data, meta = load_split(name, seed, ratio, num_transform=num_transform)
    results = []

    def ok(label, cond, detail=''):
        results.append(bool(cond))
        if verbose:
            print(f"  [{'OK ' if cond else 'GAGAL'}] {label}{(' — ' + detail) if detail else ''}")

    if verbose:
        print(f'\n=== {name} | seed {seed} | ratio {ratio} ===')
        print(f"  kolom matriks: {meta['n_num']} numerik + {sum(meta['cat_bits'])} bit kategorik "
              f"{meta['cat_bits']} + {len(meta['y_cols'])} y ({meta['task']}) = {meta['dim']}")

    n_feat = meta['n_num'] + len(meta['cat_cols'])
    for part in PARTS:
        d = data[part]
        n = len(d['df'])
        ok(f'{part}: ukuran matriks & mask cocok',
           d['X_true'].shape == d['mask'].shape == (n, meta['dim']), f"{d['X_true'].shape}")

    # proporsi missing fitur
    props = {p: data[p]['mask_feat'].mean() for p in PARTS}
    ok('proporsi missing fitur ≈ ratio',
       all(abs(v - ratio) < 0.02 for v in props.values()),
       ', '.join(f'{p} {v:.3f}' for p, v in props.items()))

    # mask y (regresi: 1 kolom; klasifikasi: blok one-hot K kolom)
    yi = meta['y_cols']
    ok('y train selalu teramati (mask False)', not data['train']['mask'][:, yi].any())
    ok('y val & test ditebak (mask True)',
       data['val']['mask'][:, yi].all() and data['test']['mask'][:, yi].all())
    ok('y tidak bocor ke input val/test (X_obs = 0)',
       (data['val']['X_obs'][:, yi] == 0).all() and (data['test']['X_obs'][:, yi] == 0).all())
    if meta['task'] == 'classification':
        Yt = data['train']['X_true'][:, yi]
        ok('one-hot y: tepat satu +0.5 per baris, sisanya -0.5',
           np.isin(Yt, [-0.5, 0.5]).all() and ((Yt == 0.5).sum(1) == 1).all(),
           f"K = {len(yi)} kelas {meta['classes']}")

    # bit kategorik dalam satu kolom punya mask yang sama
    same = True
    start = meta['n_num']
    for bits in meta['cat_bits']:
        blk = data['train']['mask'][:, start:start + bits]
        same &= bool((blk == blk[:, :1]).all())
        start += bits
    ok('mask bit kategorik seragam per kolom', same)

    # X_obs: sel tersembunyi = 0, tidak ada NaN (skala model)
    for part in PARTS:
        d = data[part]
        ok(f'{part}: X_obs tanpa NaN dan sel tersembunyi = 0',
           not np.isnan(d['X_obs']).any() and (d['X_obs'][d['mask']] == 0).all())

    # standardisasi: nilai teramati di train punya mean ≈ 0, std ≈ 0.5
    Xt, mt = data['train']['X_true'], data['train']['mask']
    obs = ~mt
    mu = np.array([Xt[obs[:, j], j].mean() for j in range(Xt.shape[1])])
    sd = np.array([Xt[obs[:, j], j].std() for j in range(Xt.shape[1])])
    nonconst = meta['std'] != 1.0
    zcols = np.arange(Xt.shape[1]) < meta['y_idx'] if meta['task'] == 'classification' \
        else np.ones(Xt.shape[1], dtype=bool)                  # blok one-hot tidak distandardisasi
    ok('standardisasi train: mean ≈ 0, std ≈ 0.5',
       np.allclose(mu[zcols], 0, atol=1e-6) and np.allclose(sd[nonconst & zcols], 0.5, atol=1e-6),
       f'max|mean| {np.abs(mu[zcols]).max():.1e}')

    # roundtrip: matriks -> skala asli harus sama dengan data asli
    for part in PARTS:
        d = data[part]
        back = inverse_transform(d['X_true'], meta)
        df = d['df']
        num_ok = np.allclose(back[meta['num_cols']].to_numpy(float),
                             df[meta['num_cols']].to_numpy(float), rtol=1e-6, atol=1e-6, equal_nan=True)
        cat_ok = all((back[c].astype(str).values == df[c].astype(str).values).all() for c in meta['cat_cols'])
        if meta['task'] == 'classification':
            y_ok = (back[meta['target']].astype(str).values == df[meta['target']].astype(str).values).all()
        else:
            y_ok = np.allclose(back[meta['target']].to_numpy(float), df[meta['target']].to_numpy(float),
                               rtol=1e-6, atol=1e-6)
        ok(f'{part}: kembali ke skala asli (numerik, kategorik, y)', num_ok and cat_ok and y_ok)

    # transformasi fitur numerik: skala model -> skala asli harus kembali ke nilai asli (sel teramati train)
    if meta.get('tf') is not None:
        d = data['train']
        back = model_to_eval(d['X_obs'], meta)
        o = ~d['mask'][:, :meta['n_num']]
        err = np.abs(back[:, :meta['n_num']][o] - d['X_true'][:, :meta['n_num']][o]).max()
        ok(f"transformasi {meta['num_transform']}: kembali ke skala asli (sel teramati train)", err < 1e-6,
           f'maks selisih {err:.1e}')

    # decode kode biner tidak valid -> kategori valid terdekat
    dec_ok = True
    for c in meta['cat_cols']:
        m = meta['cat_maps'][c]
        if 2 ** m['bits'] > len(m['categories']):
            res = decode_cats(np.ones((1, m['bits'])), [c], {c: m})[c][0]
            dec_ok &= res in m['categories']
    ok('kode biner tidak valid dipetakan ke kategori valid', dec_ok)

    # urutan baris mask cocok dengan urutan split
    idx = np.load(f"{DATA_DIR}/{name}/splits/seed{seed}/train_idx.npy")
    ok('urutan baris train cocok dengan train_idx.npy', len(idx) == len(data['train']['df']))

    if verbose:
        print(f"  Hasil: {sum(results)}/{len(results)} cek OK")
    return all(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cek pemuat data Paper 2')
    parser.add_argument('--dataset', type=str, default='bike')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0.3)
    parser.add_argument('--all', action='store_true', help='cek semua dataset x seed x ratio')
    parser.add_argument('--num_transform', default='none', choices=['none', 'quantile', 'yeojohnson'])
    args = parser.parse_args()

    if not args.all:
        check(args.dataset, args.seed, args.ratio, num_transform=args.num_transform)
    else:
        names = [f[:-5] for f in sorted(os.listdir(f'{DATA_DIR}/Info')) if f.endswith('.json')]
        total, gagal = 0, []
        for name in names:
            for seed in range(5):
                for ratio in [0.0, 0.1, 0.2, 0.3, 0.4]:
                    total += 1
                    if not check(name, seed, ratio, verbose=False):
                        gagal.append((name, seed, ratio))
        print(f'{total - len(gagal)}/{total} kombinasi lolos semua cek')
        for g in gagal:
            print('  GAGAL:', g)