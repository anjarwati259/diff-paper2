"""
Download, split, dan mask MAR untuk Paper 2 (DiffPuter untuk prediksi target regresi).

Mengikuti gaya download_and_process.py milik DiffPuter:
  1. Unduh zip dari UCI lalu unzip ke datasets/{nama}/
  2. Proses menjadi datasets/{nama}/data.csv (target selalu kolom TERAKHIR)
     + datasets/Info/{nama}.json (indeks kolom numerik, kategorik, target)
  3. Split train/val/test 70/10/20 untuk seed 0-4
     -> datasets/{nama}/splits/seed{k}/{train,val,test}_idx.npy (+ CSV)
  4. Mask missing MAR (5 mask = 5 seed; mask ke-k dipasangkan dengan split ke-k)
     untuk ratio 10%, 20%, 30%, 40%
     -> datasets/{nama}/masks/MAR{ratio}/seed{k}/{train,val,test}_mask.npy + mekanisme.json

Bedanya dengan DiffPuter:
  - Kolom target IKUT disimpan dan diletakkan di kolom terakhir.
  - Ada data validasi (70/10/20), dan split diulang untuk 5 seed.
  - Baris dengan nilai kosong asli TIDAK dibuang.
  - Target disimpan dalam skala asli; log-transform dilakukan nanti saat memuat data
    (lihat field "log_target" di Info JSON).
  - Mask hanya untuk kolom FITUR (target tidak pernah di-mask dan bukan pemicu MAR).
  - Mekanisme MAR (kolom selalu teramati, bobot, intercept) dibuat SEKALI dari data train,
    lalu dipakai juga untuk val dan test. Di DiffPuter, MAR_mask dipanggil terpisah
    untuk train dan test sehingga mekanismenya berbeda.

Pemakaian:
  python download_and_split.py                 # semua dataset (news, bike, shoppers)
  python download_and_split.py --data bike     # satu dataset saja
  python download_and_process.py --data shoppers   # klasifikasi biner (task_type = classification)
  python download_and_split.py --no_csv        # simpan indeks split saja, tanpa CSV
  python download_and_split.py --only_mask     # buat ulang mask saja (data & split sudah ada)
  python download_and_split.py --ratios 0.3    # ratio tertentu saja
"""

import os
import json
import zipfile
import argparse
from urllib import request

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit

DATA_DIR = 'datasets'

NAME_URL_DICT_UCI = {
    # 'news': 'https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip',
    # 'bike': 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
    # klasifikasi (sama dengan repo DiffPuter)
    'shoppers': 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip',
}

SPLIT_RATIO = (0.7, 0.1, 0.2)       # train / val / test
SEEDS = [0, 1, 2, 3, 4]             # 5 split = 5 mask (dipasangkan per seed)
MAR_RATIOS = [0.3]   # missing ratio 0% tidak perlu mask


# --------------------------------------------------------------------------
# Download (sama seperti DiffPuter)
# --------------------------------------------------------------------------
def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):
    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    zip_path = f'{save_dir}/{name}.zip'

    if not os.path.exists(zip_path):
        os.makedirs(save_dir, exist_ok=True)
        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, zip_path)
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        unzip_file(zip_path, save_dir)
        print(f'Finish unzipping {name}.')
    else:
        print('Already downloaded.')


# --------------------------------------------------------------------------
# Proses tiap dataset -> data.csv + Info JSON
# --------------------------------------------------------------------------
def save_data_and_info(name, df, num_cols, cat_cols, target, log_target, extra=None,
                       task_type='regression'):
    """Susun kolom: numerik, kategorik, lalu target (paling akhir).
    task_type: 'regression' atau 'classification' (dibaca data_loader.py)."""
    df = df[num_cols + cat_cols + [target]].reset_index(drop=True)

    data_path = f'{DATA_DIR}/{name}/data.csv'
    df.to_csv(data_path, index=False)

    n_num, n_cat = len(num_cols), len(cat_cols)
    info = {
        'name': name,
        'num_col_idx': list(range(n_num)),
        'cat_col_idx': list(range(n_num, n_num + n_cat)),
        'target_col_idx': [n_num + n_cat],
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'target': target,
        'task_type': task_type,
        'log_target': log_target,
        'n_rows': int(df.shape[0]),
        'source': NAME_URL_DICT_UCI.get(name, ''),
    }
    if extra:
        info.update(extra)

    os.makedirs(f'{DATA_DIR}/Info', exist_ok=True)
    with open(f'{DATA_DIR}/Info/{name}.json', 'w') as f:
        json.dump(info, f, indent=4)

    print(f'{name}: {df.shape[0]} baris, {n_num} numerik, {n_cat} kategorik, target = {target} ({task_type})')
    if task_type == 'classification':
        print(f'  proporsi kelas: {df[target].value_counts(normalize=True).sort_index().round(4).to_dict()}')
    print(f'  NaN per kolom (total): {int(df.isna().sum().sum())}')
    print(f'  Disimpan di {data_path} dan {DATA_DIR}/Info/{name}.json')
    return df


def find_file(root, filename):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f'{filename} tidak ditemukan di {root}')


def process_news():
    """
    Online News Popularity (UCI 332), 39.644 baris.
    Mengikuti process_news DiffPuter (one-hot data_channel & weekday digabung jadi 2 kategorik),
    ditambah: buang timedelta, target shares di kolom terakhir. Semua baris dipakai (tanpa sampel).
    """
    path = find_file(f'{DATA_DIR}/news', 'OnlineNewsPopularity.csv')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()          # nama kolom asli diawali spasi

    df = df.drop(columns=['url', 'timedelta'])   # url = ID, timedelta = umur artikel (tidak prediktif)

    channel_cols = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus',
                    'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']
    weekday_cols = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday',
                    'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']

    ch = df[channel_cols].astype(int).to_numpy()
    # artikel tanpa channel diberi kategori sendiri (6 = 'other');
    # di DiffPuter asli artikel seperti ini ikut masuk kategori 0 (lifestyle) karena argmax
    data_channel = np.where(ch.sum(axis=1) == 0, len(channel_cols), ch.argmax(axis=1))
    weekday = df[weekday_cols].astype(int).to_numpy().argmax(axis=1)

    # is_weekend dibuang karena sudah tercakup oleh weekday
    df = df.drop(columns=channel_cols + weekday_cols + ['is_weekend'])
    df['data_channel'] = data_channel
    df['weekday'] = weekday

    target = 'shares'
    cat_cols = ['data_channel', 'weekday']
    num_cols = [c for c in df.columns if c not in cat_cols + [target]]

    return save_data_and_info('news', df, num_cols, cat_cols, target, log_target=True)


def process_bike():
    """
    Bike Sharing (UCI 275), versi per jam (hour.csv), 17.379 baris.
    Buang instant (ID), dteday (tanggal; sudah diwakili yr, mnth, weekday),
    casual dan registered (jumlah keduanya = cnt, jadi bocor ke target).
    """
    path = find_file(f'{DATA_DIR}/bike', 'hour.csv')
    df = pd.read_csv(path)
    df = df.drop(columns=['instant', 'dteday', 'casual', 'registered'])

    target = 'cnt'
    cat_cols = ['season', 'holiday', 'weekday', 'workingday', 'weathersit']
    num_cols = ['yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed']

    return save_data_and_info('bike', df, num_cols, cat_cols, target, log_target=True)


def process_shoppers():
    """
    Online Shoppers Purchasing Intention (UCI 468), 12.330 baris, KLASIFIKASI biner.
    Kolom sama dengan repo DiffPuter (Info/shoppers.json: 10 numerik, 7 kategorik, target Revenue).
    Target Revenue (True/False) disimpan sebagai 0/1. Tidak ada baris yang dibuang
    (DiffPuter memakai dropna(); data ini memang tidak punya NaN).
    """
    path = find_file(f'{DATA_DIR}/shoppers', 'online_shoppers_intention.csv')
    df = pd.read_csv(path)

    num_cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
                'SpecialDay']
    cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
    target = 'Revenue'
    df[target] = df[target].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0}).astype(int)
    df['Weekend'] = df['Weekend'].astype(str).str.strip().str.lower().map({'true': 1, 'false': 0})

    return save_data_and_info('shoppers', df, num_cols, cat_cols, target, log_target=False,
                              task_type='classification')


# --------------------------------------------------------------------------
# Split train / val / test
# --------------------------------------------------------------------------
def make_splits(name, ratio=SPLIT_RATIO, seeds=SEEDS, save_csv=True):
    data_path = f'{DATA_DIR}/{name}/data.csv'
    df = pd.read_csv(data_path)
    n = df.shape[0]

    n_train = int(round(n * ratio[0]))
    n_val = int(round(n * ratio[1]))
    n_test = n - n_train - n_val

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        parts = {
            'train': np.sort(idx[:n_train]),
            'val': np.sort(idx[n_train:n_train + n_val]),
            'test': np.sort(idx[n_train + n_val:]),
        }
        out_dir = f'{DATA_DIR}/{name}/splits/seed{seed}'
        os.makedirs(out_dir, exist_ok=True)
        for part, part_idx in parts.items():
            np.save(f'{out_dir}/{part}_idx.npy', part_idx)
            if save_csv:
                df.iloc[part_idx].to_csv(f'{out_dir}/{part}.csv', index=False)

        # cek: tidak ada indeks yang tumpang tindih dan semua baris terpakai
        all_idx = np.concatenate(list(parts.values()))
        assert len(np.unique(all_idx)) == n, 'Split tidak valid'

    print(f'Split {name}: train {n_train}, val {n_val}, test {n_test} '
          f'untuk seed {seeds} -> {DATA_DIR}/{name}/splits/')


# --------------------------------------------------------------------------
# Mask MAR (berdasarkan MAR_mask, pick_coeffs, fit_intercepts di DiffPuter)
# --------------------------------------------------------------------------
def sigmoid(z):
    return expit(z)                               # versi stabil (tidak overflow untuk nilai besar)


def feature_matrix(df, info):
    """Matriks fitur saja (tanpa target), dalam urutan kolom data.csv.
    Kategorik dipakai sebagai kode angka, seperti generate_mask DiffPuter."""
    cols = info['num_cols'] + info['cat_cols']
    X = df[cols].copy()
    for c in info['cat_cols']:
        codes = pd.factorize(X[c], sort=True)[0].astype(np.float64)
        codes[codes < 0] = np.nan                 # kategori kosong tetap NaN (bukan kode -1)
        X[c] = codes
    return X.to_numpy(dtype=np.float64), cols


def build_mar_mechanism(X_train, ratio, rng):
    """
    Bentuk mekanisme MAR dari data train.
      - Pilih sebagian kolom yang SELALU teramati (p_obs = 30%, atau 10% jika ratio > 30%).
      - Kolom lain hilang dengan peluang sigmoid(X_obs @ W + b), W acak, b diatur
        dengan bisection agar proporsi missing tepat.
    Peluang di kolom yang bisa hilang disesuaikan supaya proporsi missing di SELURUH sel fitur = ratio.
    """
    n, d = X_train.shape
    p_obs = 0.3 if ratio <= 0.3 else 0.1
    d_obs = max(int(p_obs * d), 1)
    d_na = d - d_obs
    p_na = ratio * d / d_na                       # supaya total sel hilang = ratio
    if p_na >= 1:
        raise ValueError(f'ratio {ratio} terlalu besar untuk {d} kolom')

    idxs_obs = np.sort(rng.choice(d, d_obs, replace=False))
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # standardisasi kolom pemicu dengan statistik train (agar kolom berskala besar tidak mendominasi)
    mu = np.nanmean(X_train[:, idxs_obs], axis=0)
    sd = np.nanstd(X_train[:, idxs_obs], axis=0)
    sd[sd == 0] = 1.0
    Z = np.nan_to_num((X_train[:, idxs_obs] - mu) / sd)

    coeffs = rng.standard_normal((d_obs, d_na))
    Wx = Z @ coeffs
    std_wx = Wx.std(axis=0, ddof=1)
    std_wx[std_wx == 0] = 1.0
    coeffs = coeffs / std_wx                      # W^T x bervariansi 1 (pick_coeffs)

    intercepts = np.zeros(d_na)
    for j in range(d_na):                         # fit_intercepts
        zj = Z @ coeffs[:, j]
        f = lambda b: sigmoid(zj + b).mean() - p_na
        intercepts[j] = optimize.bisect(f, -500, 500)

    return {
        'ratio': ratio, 'p_obs': p_obs, 'p_na': p_na,
        'idxs_obs': idxs_obs.tolist(), 'idxs_nas': idxs_nas.tolist(),
        'mu': mu.tolist(), 'sd': sd.tolist(),
        'coeffs': coeffs.tolist(), 'intercepts': intercepts.tolist(),
    }


def apply_mar_mechanism(X, mech, rng):
    """Terapkan mekanisme yang sama ke train, val, atau test. True = hilang."""
    n, d = X.shape
    idxs_obs = np.array(mech['idxs_obs'])
    idxs_nas = np.array(mech['idxs_nas'])
    Z = np.nan_to_num((X[:, idxs_obs] - np.array(mech['mu'])) / np.array(mech['sd']))
    ps = sigmoid(Z @ np.array(mech['coeffs']) + np.array(mech['intercepts']))
    mask = np.zeros((n, d), dtype=bool)
    mask[:, idxs_nas] = rng.random((n, len(idxs_nas))) < ps
    mask |= np.isnan(X)                           # nilai yang memang kosong dari awal ikut ditandai
    return mask


def make_mar_masks(name, ratios=MAR_RATIOS, seeds=SEEDS):
    df = pd.read_csv(f'{DATA_DIR}/{name}/data.csv')
    with open(f'{DATA_DIR}/Info/{name}.json') as f:
        info = json.load(f)
    X, feat_cols = feature_matrix(df, info)

    ratios = [r for r in ratios if r > 0]         # ratio 0% tidak perlu mask
    print(f'Mask MAR {name}: {X.shape[1]} kolom fitur, ratio {ratios}, seed {seeds}')
    for ratio in ratios:
        tag = f'MAR{int(round(ratio * 100)):02d}'
        for seed in seeds:
            split_dir = f'{DATA_DIR}/{name}/splits/seed{seed}'
            idx = {p: np.load(f'{split_dir}/{p}_idx.npy') for p in ['train', 'val', 'test']}

            # seed berbeda untuk tiap (seed, ratio), tetap bisa direproduksi
            rng = np.random.default_rng(10_000 * seed + int(round(ratio * 100)))
            mech = build_mar_mechanism(X[idx['train']], ratio, rng)

            out_dir = f'{DATA_DIR}/{name}/masks/{tag}/seed{seed}'
            os.makedirs(out_dir, exist_ok=True)
            actual = {}
            for part in ['train', 'val', 'test']:
                m = apply_mar_mechanism(X[idx[part]], mech, rng)
                np.save(f'{out_dir}/{part}_mask.npy', m)   # baris mengikuti urutan {part}_idx.npy
                actual[part] = float(m.mean())

            mech.update({
                'mekanisme': 'MAR', 'seed': seed, 'feature_cols': feat_cols,
                'kolom_selalu_teramati': [feat_cols[i] for i in mech['idxs_obs']],
                'proporsi_missing_aktual': actual,
            })
            with open(f'{out_dir}/mekanisme.json', 'w') as f:
                json.dump(mech, f, indent=2)

        print(f'  {tag}: contoh seed {seeds[-1]} proporsi aktual '
              f"train {actual['train']:.3f}, val {actual['val']:.3f}, test {actual['test']:.3f}")
    print(f'  Disimpan di {DATA_DIR}/{name}/masks/')


# --------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download, split, dan mask MAR dataset Paper 2')
    parser.add_argument('--data', type=str, default='all', help="'all' atau salah satu: news, bike, shoppers")
    parser.add_argument('--no_download', action='store_true', help='lewati unduhan (zip sudah ada)')
    parser.add_argument('--no_csv', action='store_true', help='simpan indeks split saja, tanpa CSV')
    parser.add_argument('--only_mask', action='store_true', help='buat mask saja (data dan split sudah ada)')
    parser.add_argument('--ratios', type=float, nargs='+', default=MAR_RATIOS, help='missing ratio MAR')
    args = parser.parse_args()

    names = list(NAME_URL_DICT_UCI.keys()) if args.data == 'all' else [args.data]

    for name in names:
        if not args.only_mask:
            if not args.no_download:
                download_from_uci(name)
            globals()[f'process_{name}']()
            make_splits(name, save_csv=not args.no_csv)
        make_mar_masks(name, ratios=args.ratios)
        print()