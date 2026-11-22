"""
BASELINE yang menangani missing value LANGSUNG saat prediksi (tanpa tahap imputasi terpisah), untuk
REGRESI dan KLASIFIKASI. Split, mask, dan metrik SAMA dengan metode usulan & run_xgboost_nan.py.
(XGBoost NaN bawaan ada di skrip sendiri: run_xgboost_nan.py.)

  --method   keluarga                      kode yang dipakai                                       acuan
  lightgbm   gradient boosting             paket pip lightgbm (NaN bawaan)                         Ke dkk., NeurIPS 2017
  catboost   gradient boosting simetris    paket pip catboost (NaN = 'Min', kategorik bawaan)      Prokhorenkova dkk., NeurIPS 2018
  grape      GNN graf bipartit             baselines_eksternal/GRAPE (kode resmi, alur gnn_y.py)   You dkk., NeurIPS 2020
  neumiss    jaringan Neumann + MLP        baselines_eksternal/NeuMiss (NeuMissBlock resmi) +      Le Morvan dkk., NeurIPS 2021
                                           setting & grid 144 kombinasi dari paper, dipilih di val
  supmiwae   VAE + classifier/regressor    baselines_eksternal/supMIWAE (port PyTorch kode resmi)  Ipsen dkk., ICLR 2022

Aturan sama untuk semua:
  - Data: data_loader.load_split (split 70/10/20, mask MAR yang sama, seed = split + mask).
  - Model dilatih di TRAIN; VAL hanya untuk early stopping / pemilihan epoch, ambang F1, dan T (Brier).
  - Hyperparameter default dari kode resmi / paket (tanpa tuning), dicatat di metrics.json.
  - Klasifikasi: metrics_cls.cls_metrics (probabilitas mentah) + ambang F1 macro dari val + TS (Brier).
  - Regresi: SKALA STANDAR-LOG (z-score log1p(y) statistik train = data_loader.y_true_z), tanpa nilai asli.
    LightGBM/CatBoost dilatih pada y skala itu; GRAPE/NeuMiss/supMIWAE pada kolom y matriks lalu inverse_y.
  - Regresi: RMSE, MAE, R2 di skala standar-log; CRPS bila metode memberi distribusi (LightGBM/CatBoost: kuantil,
    supMIWAE: sampel), GRAPE & NeuMiss prediksi titik saja -> CRPS = NaN.
  - Imputasi fitur (MAE/RMSE/ACC, sama dengan usulan) dilaporkan untuk supMIWAE (punya isian eksplisit).

Kebutuhan: pip install lightgbm catboost torch_geometric  (torch_scatter TIDAK wajib; ada pengganti otomatis)

Pemakaian (dari folder proyek):
  python run_baseline.py --method lightgbm --dataset bike --ratio 0.3 --seed 0
  python run_baseline.py --method grape    --dataset shoppers --ratio 0.3 --seed 0
Hasil: {folder run_baseline.py}/results_zlog/{method}/{dataset}/MAR{rr}/seed{k}/{method}_{metrics.json, val.csv, test.csv} + result.txt (append)
"""

import os
import sys
import json
import time
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- cari folder proyek (yang berisi data_loader.py): folder skrip ini atau folder induknya.
#      Skrip boleh diletakkan di folder proyek ATAU di baselines_eksternal/; data, mask, dan hasil
#      selalu dibaca/ditulis relatif terhadap folder proyek.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = next((d for d in (_HERE, os.path.dirname(_HERE)) if os.path.isfile(os.path.join(d, 'data_loader.py'))), None)
if PROJECT_DIR is None:
    raise FileNotFoundError('data_loader.py tidak ditemukan di folder skrip maupun folder induknya')
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from data_loader import load_split, inverse_y, y_true_index, y_true_z, load_ids
from run_diffputer import imputation_metrics, y_metrics, crps_fair
import metrics_cls as mc

METHODS = ['lightgbm', 'catboost', 'grape', 'neumiss', 'supmiwae']


# =============================================================================================
# Data
# =============================================================================================
def frames_native(data, meta):
    """Fitur skala asli, sel di-mask = NaN, kategorik = pandas category (kategori dari train)."""
    num_cols, cat_cols = list(meta['num_cols']), list(meta['cat_cols'])
    feat = num_cols + cat_cols
    fr = {}
    for part, d in data.items():
        X = d['df'][feat].copy().mask(np.asarray(d['mask_feat'], dtype=bool))
        for c in num_cols:
            X[c] = X[c].astype(np.float64)
        fr[part] = X
    for c in cat_cols:
        cats = sorted(pd.Series(data['train']['df'][c]).dropna().unique().tolist(), key=str)
        for part in fr:
            fr[part][c] = pd.Categorical(fr[part][c], categories=cats)
    return fr, feat


def matrix_parts(data, meta):
    """Matriks terstandardisasi data_loader (numerik + bit kategorik, TANPA kolom y) + mask fitur (1 = hilang)."""
    yi = meta['y_idx']
    out = {}
    for part, d in data.items():
        X = d['X_obs'][:, :yi].astype(np.float32)
        M = d['mask'][:, :yi].astype(np.float32)
        out[part] = (X, M)
    return out


def targets(data, meta):
    """Regresi: y terstandardisasi (skala model) + y asli. Klasifikasi: indeks kelas."""
    if meta['task'] == 'classification':
        return {p: y_true_index(data[p]['df'], meta) for p in data}, None
    yi = meta['y_idx']
    return ({p: data[p]['X_true'][:, yi].astype(np.float32) for p in data},
            {p: y_true_z(data[p]['df'], meta) for p in data})                     # skala standar-log


# =============================================================================================
# Metrik
# =============================================================================================
def crps_quantile(Q, y):
    N = Q.shape[0]
    t1 = np.abs(Q - y[None, :]).mean(0)
    t2 = np.abs(Q[:, None, :] - Q[None, :, :]).sum((0, 1)) / (2 * N * N)
    return float((t1 - t2).mean())


def reg_metrics(y, point, samples=None, quantiles=None):
    """y, point skala standar-log. samples [N, n] (iid) atau quantiles [N, n] (level k/(N+1))."""
    if samples is not None:
        m = y_metrics(y, samples)
        m.update({'rmse': float(np.sqrt(((y - point) ** 2).mean())), 'mae': float(np.abs(y - point).mean()),
                  'r2': float(1 - ((y - point) ** 2).sum() / ((y - y.mean()) ** 2).sum())})
        return m
    m = {'rmse': float(np.sqrt(((y - point) ** 2).mean())), 'mae': float(np.abs(y - point).mean()),
         'r2': float(1 - ((y - point) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
         'crps': float('nan'), 'coverage_minmax': float('nan'), 'coverage_p5_p95': float('nan')}
    if quantiles is not None:
        Q = np.sort(quantiles, axis=0)
        m['crps'] = crps_quantile(Q, y)
        m['coverage_minmax'] = float(((y >= Q[0]) & (y <= Q[-1])).mean())
        m['coverage_minmax_ideal'] = (Q.shape[0] - 1) / (Q.shape[0] + 1)
    return m


def q_levels(n=10):
    return [k / (n + 1) for k in range(1, n + 1)]


# =============================================================================================
# Pohon: LightGBM & CatBoost (default + early stopping di val, kategorik bawaan)
# =============================================================================================
def fit_lightgbm(task, fr, ytr, yva, K, a):
    import lightgbm as lgb
    base = dict(learning_rate=0.1, num_leaves=31, min_data_in_leaf=20, feature_fraction=1.0, bagging_fraction=1.0,
                seed=a.seed, verbose=-1, use_missing=True, zero_as_missing=False, num_threads=a.n_jobs)
    cb = [lgb.early_stopping(100, verbose=False)]
    if task == 'classification':
        p = {**base, 'objective': 'binary' if K == 2 else 'multiclass', **({} if K == 2 else {'num_class': K})}
        bst = lgb.train(p, lgb.Dataset(fr['train'], ytr), num_boost_round=5000,
                        valid_sets=[lgb.Dataset(fr['val'], yva)], callbacks=cb)

        def proba(X):
            pr = bst.predict(X, num_iteration=bst.best_iteration)
            return np.stack([1 - pr, pr], 1) if K == 2 else pr
        return {'proba': proba, 'info': {'params': p, 'n_trees': bst.best_iteration}}
    p = {**base, 'objective': 'regression'}
    t0 = time.time()
    bst = lgb.train(p, lgb.Dataset(fr['train'], ytr), 5000, valid_sets=[lgb.Dataset(fr['val'], yva)], callbacks=cb)
    print(f'  model titik selesai: {bst.best_iteration} pohon, {time.time() - t0:.0f}s', flush=True)
    qm = []
    for k, al in enumerate(q_levels(a.num_q), 1):
        t0 = time.time()
        pq = {**base, 'objective': 'quantile', 'alpha': al}
        qm.append(lgb.train(pq, lgb.Dataset(fr['train'], ytr), 5000, valid_sets=[lgb.Dataset(fr['val'], yva)],
                            callbacks=[lgb.early_stopping(100, verbose=False)]))
        print(f'  model kuantil {k}/{a.num_q} (alpha {al:.3f}) selesai: {qm[-1].best_iteration} pohon, '
              f'{time.time() - t0:.0f}s', flush=True)
    return {'point': lambda X: bst.predict(X, num_iteration=bst.best_iteration),
            'quant': lambda X: np.stack([m.predict(X, num_iteration=m.best_iteration) for m in qm], 0),
            'info': {'params': p, 'n_trees': bst.best_iteration}}


def fit_catboost(task, fr, ytr, yva, K, a, cat_cols):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    def pool(X, y=None):
        X = X.copy()
        for c in cat_cols:                                   # CatBoost: kategorik sebagai string, hilang = 'NA'
            X[c] = X[c].astype(object).where(X[c].notna(), 'NA').astype(str)
        return Pool(X, y, cat_features=cat_cols)
    common = dict(iterations=5000, learning_rate=None, depth=6, random_seed=a.seed, verbose=False,
                  nan_mode='Min', early_stopping_rounds=100, thread_count=a.n_jobs)
    if task == 'classification':
        m = CatBoostClassifier(**common, loss_function='Logloss' if K == 2 else 'MultiClass')
        m.fit(pool(fr['train'], ytr), eval_set=pool(fr['val'], yva), use_best_model=True)
        return {'proba': lambda X: m.predict_proba(pool(X)), 'info': {'n_trees': int(m.get_best_iteration() or 0) + 1}}
    t0 = time.time()
    m = CatBoostRegressor(**common, loss_function='RMSE')
    m.fit(pool(fr['train'], ytr), eval_set=pool(fr['val'], yva), use_best_model=True)
    print(f'  model titik selesai: {time.time() - t0:.0f}s', flush=True)
    al = ','.join(f'{x:.6f}' for x in q_levels(a.num_q))
    mq = CatBoostRegressor(**common, loss_function=f'MultiQuantile:alpha={al}')
    mq.fit(pool(fr['train'], ytr), eval_set=pool(fr['val'], yva), use_best_model=True)
    return {'point': lambda X: m.predict(pool(X)), 'quant': lambda X: np.asarray(mq.predict(pool(X))).T,
            'info': {'n_trees': int(m.get_best_iteration() or 0) + 1}}


# =============================================================================================
# Kode resmi di baselines_eksternal/
# =============================================================================================
EXT_DIR = os.path.join(PROJECT_DIR, 'baselines_eksternal')


def _add_path(sub, flat_file=None):
    """Tambahkan baselines_eksternal/{sub} ke sys.path. Bila subfolder tidak ada tetapi file-nya diletakkan
    langsung di baselines_eksternal/ (flat_file), folder itu yang dipakai."""
    p = os.path.join(EXT_DIR, sub)
    if not os.path.isdir(p):
        if flat_file and os.path.isfile(os.path.join(EXT_DIR, flat_file)):
            p = EXT_DIR
        else:
            raise FileNotFoundError(f'Folder {p} tidak ada. Letakkan kode di baselines_eksternal/{sub}/')
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def head_out(task, K):
    return K if task == 'classification' else 1


def sup_loss(task, out, y):
    return F.cross_entropy(out, y) if task == 'classification' else F.mse_loss(out[:, 0], y)


# ---------------------------------------------------------------------------------------------
# NeuMiss+MLP — setting PAPER Le Morvan dkk., NeurIPS 2021 ("What's a good imputation to predict with
#   missing values?"), kode eksperimen github.com/marineLM/Impute_then_Regress (launch_all.py,
#   NeuMiss_accelerated_with_init.py). Blok Neumann = kode resmi baselines_eksternal/NeuMiss (NeuMissBlock,
#   bobot dibagi + skip connection = mode 'shared' + residual_connection=True di paper).
#   Paper:  kedalaman Neumann 20; MLP setelah blok: mlp_depth ∈ {1,2,5} lapisan ReLU, lebar = width_factor × d,
#           width_factor ∈ {1,5,10}; lr ∈ {1e-2,5e-3,1e-3,5e-4}; weight decay ∈ {1e-6,1e-5,1e-4,1e-3}
#           (mu & bias keluaran tanpa weight decay); Adam, batch 100, maks 1000 epoch, ReduceLROnPlateau
#           (factor 0.2, patience 10, threshold 1e-4), berhenti bila lr < 1e-4, early stopping patience 12
#           (delta 1e-4) -> bobot val terbaik; inisialisasi 'uniform' (mu = 0, W Neumann xavier_uniform gain 0.5,
#           W MLP xavier_uniform gain sqrt(2), bias 0, bobot keluaran U(±sqrt(12/(in+1)))).
#           144 kombinasi; kombinasi TERBAIK dipilih di VAL (paper: R2 val = MSE val).
#   Penyesuaian: klasifikasi -> keluaran K logit + cross-entropy, dipilih dengan CE val (paper hanya regresi).
# ---------------------------------------------------------------------------------------------
NEUMISS_GRID = {
    'paper': dict(depth=[20], mlp_depth=[1, 2, 5], width_factor=[1, 5, 10],
                  lr=[1e-2, 5e-3, 1e-3, 5e-4], weight_decay=[1e-6, 1e-5, 1e-4, 1e-3]),
    'kecil': dict(depth=[20], mlp_depth=[1, 2], width_factor=[5, 10], lr=[1e-3, 5e-3], weight_decay=[1e-5, 1e-4]),
}


def build_neumiss(d, out, depth, mlp_depth, width_factor):
    p = _add_path('NeuMiss', flat_file='NeuMissBlock.py')
    if os.path.isfile(os.path.join(p, 'NeuMissBlock.py')):            # file diletakkan tanpa subfolder neumiss/
        from NeuMissBlock import NeuMissBlock
    else:
        from neumiss.NeuMissBlock import NeuMissBlock
    blk = NeuMissBlock(d, depth)
    nn.init.zeros_(blk.mu)                                               # paper: mu = 0
    nn.init.xavier_uniform_(blk.linear.weight, gain=0.5)
    w = width_factor * d
    layers, i = [blk], d
    for _ in range(mlp_depth):
        lin = nn.Linear(i, w)
        nn.init.xavier_uniform_(lin.weight, gain=math.sqrt(2)); nn.init.zeros_(lin.bias)
        layers += [lin, nn.ReLU()]
        i = w
    head = nn.Linear(i, out)
    tmp = math.sqrt(2 * 6 / (i + 1))
    nn.init.uniform_(head.weight, -tmp, tmp); nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


def _fit_neumiss_one(task, Xtr, ytr, Xva, yva, K, cfg, dev, max_epochs):
    d = Xtr.shape[1]
    model = build_neumiss(d, head_out(task, K), cfg['depth'], cfg['mlp_depth'], cfg['width_factor']).to(dev)
    last = len(model) - 1
    no_wd = {'0.mu', f'{last}.bias'}                                      # paper: 'mu' & 'b' tanpa weight decay
    g_wd = [p for n_, p in model.named_parameters() if n_ not in no_wd]
    g_no = [p for n_, p in model.named_parameters() if n_ in no_wd]
    opt = torch.optim.Adam([{'params': g_wd, 'weight_decay': cfg['weight_decay']},
                            {'params': g_no, 'weight_decay': 0.0}], lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=10, threshold=1e-4)
    best, best_state, bad, ep = float('inf'), None, 0, 0
    for ep in range(max_epochs):
        if opt.param_groups[0]['lr'] < 1e-4:                               # paper: berhenti bila lr < 1e-4
            break
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for s in range(0, len(Xtr), 100):
            i = perm[s:s + 100]
            loss = sup_loss(task, model(Xtr[i]), ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(sup_loss(task, model(Xva), yva))
        if not np.isfinite(v):
            break
        if v < best - 1e-4 or best_state is None:                          # EarlyStopping: delta 1e-4, patience 12
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
        sched.step(v)
    if best_state is None:
        return None, float('inf'), ep + 1
    model.load_state_dict(best_state); model.eval()
    return model, best, ep + 1


def run_neumiss(task, mats, Y, K, a, dev):
    import itertools
    T = lambda z, dt=torch.float32: torch.tensor(z, dtype=dt, device=dev)
    ydt = torch.long if task == 'classification' else torch.float32
    Xn = {p: np.where(mats[p][1] == 1, np.nan, mats[p][0]).astype(np.float32) for p in mats}   # NaN = hilang
    Xtr, ytr = T(Xn['train']), T(Y['train'], ydt)
    Xva, yva = T(Xn['val']), T(Y['val'], ydt)
    G = NEUMISS_GRID[a.neumiss_grid]
    keys = ['depth', 'mlp_depth', 'width_factor', 'lr', 'weight_decay']
    combos = [dict(zip(keys, c)) for c in itertools.product(*[G[k] for k in keys])]
    best_model, best_val, best_cfg, tabel = None, float('inf'), None, []
    t_all = time.time()
    for j, cfg in enumerate(combos, 1):
        t0 = time.time()
        torch.manual_seed(a.seed)
        model, v, n_ep = _fit_neumiss_one(task, Xtr, ytr, Xva, yva, K, cfg, dev, a.max_epochs)
        tabel.append({**cfg, 'val_loss': v, 'epoch': n_ep})
        tanda = ''
        if v < best_val:
            best_model, best_val, best_cfg, tanda = model, v, cfg, '  <- terbaik sejauh ini'
        print(f'  NeuMiss {j}/{len(combos)} mlp_depth={cfg["mlp_depth"]} wf={cfg["width_factor"]} lr={cfg["lr"]:g} '
              f'wd={cfg["weight_decay"]:g}: val {v:.4f} ({n_ep} epoch, {time.time() - t0:.0f}s){tanda}', flush=True)
    res = {}
    with torch.no_grad():
        for p in ('val', 'test'):
            o = best_model(T(Xn[p]))
            res[p] = torch.softmax(o, 1).cpu().numpy() if task == 'classification' else o[:, 0].cpu().numpy()
    return res, {'kode': 'baselines_eksternal/NeuMiss (NeuMissBlock resmi) + setting paper NeurIPS 2021',
                 'grid': a.neumiss_grid, 'n_kombinasi': len(combos), 'terpilih': best_cfg, 'val_terbaik': best_val,
                 'kriteria_val': 'CE val' if task == 'classification' else 'MSE val (= R2 val paper)',
                 'semua_kombinasi': tabel, 'waktu_grid_s': time.time() - t_all}


# ---------------------------------------------------------------------------------------------
# GRAPE — kode resmi baselines_eksternal/GRAPE (models/gnn_model.get_gnn, models/prediction_model.MLPNet,
#   utils/utils) dengan alur training training/gnn_y.py (train_y.py). Default = argumen train_y.py:
#   EGSAGE_EGSAGE, node/edge dim 16, edge_mode 1, aggr mean, post_hiddens [node_dim], impute & predict
#   tanpa hidden layer, Adam lr 1e-3, known 0.7 (edge dropout tiap epoch), 20000 epoch full-batch,
#   node_mode 0 (simpul fitur one-hot, simpul baris vektor 1), fitur MinMax (uci_data.get_data).
#   Graf transduktif: SEMUA baris (train/val/test) jadi simpul, edge = sel TERAMATI; label hanya train.
#   Penyesuaian: (1) val dipakai memilih epoch terbaik (resmi --valid), (2) klasifikasi: predict_model
#   keluaran K + cross-entropy (resmi hanya regresi MSE), (3) graf dibangun dengan numpy (urutan edge
#   sama dengan create_edge resmi, yang memakai loop list O(n^2) dan terlalu lambat untuk >10 ribu baris).
# ---------------------------------------------------------------------------------------------
def _shim_torch_scatter():
    """GRAPE resmi mengimpor torch_scatter.scatter_add (tidak dipakai EGSAGE). Bila paket tidak terpasang
    (sulit di Windows / torch baru), sediakan pengganti setara berbasis torch."""
    try:
        import torch_scatter  # noqa: F401
        return 'torch_scatter asli'
    except ImportError:
        import types
        mod = types.ModuleType('torch_scatter')

        def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
            dim = dim % src.dim()
            if out is None:
                size = list(src.shape)
                size[dim] = dim_size if dim_size is not None else int(index.max()) + 1
                out = torch.full(size, fill_value, dtype=src.dtype, device=src.device)
            idx = index
            while idx.dim() < src.dim():
                idx = idx.unsqueeze(-1)
            return out.scatter_add_(dim, idx.expand_as(src), src)
        mod.scatter_add = scatter_add
        sys.modules['torch_scatter'] = mod
        return 'pengganti torch_scatter (scatter_add berbasis torch)'


def run_grape(task, mats, Y, K, a, dev):
    gdir = _add_path('GRAPE')
    shim = _shim_torch_scatter()
    try:
        from models.gnn_model import get_gnn
        from models.prediction_model import MLPNet
        from utils.utils import build_optimizer, get_known_mask, mask_edge
    except ImportError as e:
        raise ImportError(f'{e}\nGRAPE resmi butuh torch_geometric: pip install torch_geometric') from e
    if not os.path.abspath(sys.modules['models.gnn_model'].__file__).startswith(os.path.abspath(gdir)):
        raise ImportError('modul "models" yang terimpor bukan milik baselines_eksternal/GRAPE (bentrok nama)')

    parts = ['train', 'val', 'test']
    X = np.concatenate([mats[p][0] for p in parts]).astype(np.float64)
    M = np.concatenate([mats[p][1] for p in parts]).astype(bool)          # True = hilang
    off = np.cumsum([0] + [len(mats[p][0]) for p in parts])
    n_row, n_col = X.shape
    # MinMax per kolom (resmi: MinMaxScaler di get_data); dihitung dari nilai TERAMATI baris train
    Xtr_o = np.where(M[:off[1]], np.nan, X[:off[1]])
    lo, hi = np.nanmin(Xtr_o, 0), np.nanmax(Xtr_o, 0)
    X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    X[M] = 0.0
    # graf persis create_node(mode 0) / create_edge / create_edge_attr resmi
    rows = np.repeat(np.arange(n_row), n_col); cols = np.tile(np.arange(n_col), n_row)
    edge_index = np.stack([np.concatenate([rows, n_row + cols]), np.concatenate([n_row + cols, rows])])
    edge_attr = np.concatenate([X.reshape(-1), X.reshape(-1)])[:, None]
    node = np.concatenate([np.ones((n_row, n_col)), np.eye(n_col)])
    known = ~M.reshape(-1)                                                 # resmi: train_edge_mask = sel teramati
    kk = np.concatenate([known, known])
    T = lambda z, dt=torch.float32: torch.tensor(z, dtype=dt, device=dev)
    x = T(node); edge_index = T(edge_index, torch.long); edge_attr = T(edge_attr)
    train_edge_index, train_edge_attr = edge_index[:, T(kk, torch.bool)], edge_attr[T(kk, torch.bool)]
    ycat = np.concatenate([Y[p] for p in parts])
    y = T(ycat, torch.long if task == 'classification' else torch.float32)
    tr = torch.zeros(n_row, dtype=torch.bool, device=dev); tr[off[0]:off[1]] = True
    va = torch.zeros(n_row, dtype=torch.bool, device=dev); va[off[1]:off[2]] = True

    g = argparse.Namespace(model_types='EGSAGE_EGSAGE', post_hiddens=None, concat_states=False, norm_embs=None,
                           aggr='mean', node_dim=a.grape_node_dim, edge_dim=a.grape_edge_dim, edge_mode=1,
                           gnn_activation='relu', impute_hiddens='', impute_activation='relu', predict_hiddens='',
                           epochs=a.grape_epochs, opt='adam', opt_scheduler='none', opt_restart=0,
                           opt_decay_step=1000, opt_decay_rate=0.9, dropout=0.0, weight_decay=0.0, lr=1e-3,
                           known=a.grape_known)
    gdata = argparse.Namespace(num_node_features=n_col, edge_attr_dim=1)
    model = get_gnn(gdata, g).to(dev)
    impute_model = MLPNet(g.node_dim * 2, 1, hidden_layer_sizes=[], hidden_activation='relu', dropout=0.0).to(dev)
    predict_model = MLPNet(n_col, head_out(task, K), hidden_layer_sizes=[], dropout=0.0).to(dev)
    scheduler, opt = build_optimizer(g, list(model.parameters()) + list(impute_model.parameters())
                                     + list(predict_model.parameters()))
    nn_ = n_row * n_col

    def forward(ei, ea):
        emb = model(x, ea, ei)
        Xh = impute_model([emb[edge_index[0, :nn_]], emb[edge_index[1, :nn_]]])
        return predict_model(torch.reshape(Xh, [n_row, n_col]))

    nets = (model, impute_model, predict_model)
    best, best_state, bad, ep_best = float('inf'), None, 0, 0
    half = train_edge_attr.shape[0] // 2
    print(f'  GRAPE: {n_row} simpul baris, {n_col} simpul fitur, {half} sel teramati (edge), device {dev}', flush=True)
    t_start = time.time()
    for ep in range(g.epochs):
        for m_ in nets:
            m_.train()
        km = get_known_mask(g.known, half).to(dev)
        kei, kea = mask_edge(train_edge_index, train_edge_attr, torch.cat((km, km), 0), True)
        opt.zero_grad()
        loss = sup_loss(task, forward(kei, kea)[tr], y[tr])
        loss.backward(); opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % a.grape_eval_every == 0 or ep == g.epochs - 1:
            for m_ in nets:
                m_.eval()
            with torch.no_grad():
                v = float(sup_loss(task, forward(train_edge_index, train_edge_attr)[va], y[va]))
            if v < best - 1e-6:
                best, bad, ep_best = v, 0, ep
                best_state = [{k: t.detach().clone() for k, t in m_.state_dict().items()} for m_ in nets]
            else:
                bad += a.grape_eval_every
            if ep % 500 == 0:
                print(f'  GRAPE epoch {ep}/{g.epochs}: loss val {v:.4f} | terbaik {best:.4f} di epoch {ep_best} | '
                      f'{time.time() - t_start:.0f}s', flush=True)
            if bad >= a.grape_patience:
                print(f'  GRAPE berhenti di epoch {ep}: {a.grape_patience} epoch tanpa perbaikan val', flush=True)
                break
    for m_, st in zip(nets, best_state):
        m_.load_state_dict(st); m_.eval()
    with torch.no_grad():
        out = forward(train_edge_index, train_edge_attr).cpu().numpy()
    res = {}
    for i, p in enumerate(parts):
        o = out[off[i]:off[i + 1]]
        res[p] = torch.softmax(torch.tensor(o), 1).numpy() if task == 'classification' else o[:, 0]
    return res, {'kode': 'baselines_eksternal/GRAPE (resmi, alur gnn_y.py)', 'torch_scatter': shim,
                 'epoch_terbaik': ep_best, 'epoch_jalan': ep + 1, 'node_dim': g.node_dim, 'edge_dim': g.edge_dim,
                 'model_types': g.model_types, 'edge_mode': 1, 'known': g.known, 'lr': g.lr}


# ---------------------------------------------------------------------------------------------
# supMIWAE — port PyTorch kode resmi: baselines_eksternal/supMIWAE/supmiwae_torch.py
# ---------------------------------------------------------------------------------------------
def run_supmiwae(task, mats, Y, K, a, dev):
    _add_path('supMIWAE', flat_file='supmiwae_torch.py')
    from supmiwae_torch import fit_supmiwae, predict_batched
    cfg = {'n_hidden': (50, 50, 50), 'n_latent': a.miwae_latent or None, 'n_samples': a.miwae_K,
           'batch_size': 100, 'eval_every': 200, 'lr_pretrain': 1e-3, 'lr_sup': 1e-4,
           'max_epochs': a.max_epochs, 'patience_eval': a.miwae_patience_eval, 'batch_pred': 256}
    S = {p: 1.0 - mats[p][1] for p in mats}                                  # 1 = teramati (s resmi)
    X = {p: mats[p][0] * S[p] for p in mats}                                 # standardisasi, hilang = 0
    model, log = fit_supmiwae(task, X['train'], S['train'], Y['train'], X['val'], S['val'], Y['val'], K, cfg, dev)
    res, imput, draws = {}, {}, {}
    for p in ('val', 'test'):
        o, imp, dr = predict_batched(model, X[p], S[p], a.miwae_K_test, 0 if task == 'classification' else a.num_q,
                                     cfg['batch_pred'])
        res[p], imput[p] = o.cpu().numpy(), imp.cpu().numpy()
        if dr is not None:
            draws[p] = dr.cpu().numpy()
    log.update({'kode': 'baselines_eksternal/supMIWAE/supmiwae_torch.py (port resmi)', 'K_train': a.miwae_K,
                'K_test': a.miwae_K_test})
    return res, imput, draws, log


# =============================================================================================
# Utama
# =============================================================================================
def main():
    ap = argparse.ArgumentParser(description='Baseline yang menangani missing value saat prediksi')
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mech', default='MAR')
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'results_zlog'),
                    help='folder hasil (bawaan: results_zlog/ di folder tempat run_baseline.py berada)')
    ap.add_argument('--device', default='auto', help="auto | cpu | cuda")
    ap.add_argument('--num_q', type=int, default=10, help='jumlah kuantil/sampel untuk CRPS regresi (= N usulan)')
    ap.add_argument('--thr_objective', default='f1_macro', choices=['f1_macro', 'f1_weighted'])
    ap.add_argument('--n_jobs', type=int, default=8,
                    help='LightGBM/CatBoost: jumlah thread CPU (semua core = lambat bila core banyak / ada run lain)')
    ap.add_argument('--max_epochs', type=int, default=1000, help='NeuMiss (paper: 1000) & supMIWAE: maks epoch')
    ap.add_argument('--neumiss_grid', default='paper', choices=list(NEUMISS_GRID),
                    help='NeuMiss: paper = 144 kombinasi persis paper; kecil = 16 kombinasi (uji cepat)')
    ap.add_argument('--grape_epochs', type=int, default=20000, help='GRAPE: epoch full-batch (train_y.py: 20000)')
    ap.add_argument('--grape_eval_every', type=int, default=20)
    ap.add_argument('--grape_patience', type=int, default=2000, help='GRAPE: epoch tanpa perbaikan val')
    ap.add_argument('--grape_node_dim', type=int, default=16)
    ap.add_argument('--grape_edge_dim', type=int, default=16)
    ap.add_argument('--grape_known', type=float, default=0.7, help='GRAPE: 1 - edge dropout')
    ap.add_argument('--miwae_K', type=int, default=50, help='supMIWAE: sampel importance saat training (resmi 50)')
    ap.add_argument('--miwae_K_test', type=int, default=1000, help='supMIWAE: sampel saat prediksi (resmi 10000)')
    ap.add_argument('--miwae_latent', type=int, default=0, help='supMIWAE: dimensi laten (0 = d - 1)')
    ap.add_argument('--miwae_patience_eval', type=int, default=20,
                    help='supMIWAE: berhenti bila skor val tidak membaik sekian evaluasi (1 evaluasi = 200 langkah)')
    ap.add_argument('--quick', action='store_true', help='uji alur cepat (hasil TIDAK untuk dilaporkan)')
    a = ap.parse_args()
    if a.quick:
        a.max_epochs, a.grape_epochs, a.grape_patience = 3, 60, 40
        a.neumiss_grid = 'kecil'
        a.miwae_K, a.miwae_K_test, a.miwae_patience_eval = 5, 20, 2
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')

    t_all = time.time()
    data, meta = load_split(a.dataset, a.seed, a.ratio, a.mech)
    task = meta['task']; K = len(meta['classes']) if task == 'classification' else 1
    Y, Yorig = targets(data, meta)
    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    pred_dir = f'{a.out_dir}/{a.method}/{a.dataset}/{tag}/seed{a.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    name = a.method + ('_quick' if a.quick else '')
    print(f'=== {a.method} | {a.dataset} ({task}{", " + str(K) + " kelas" if K > 1 else ""}) | {tag} seed{a.seed} | '
          f'device {dev} | missing train {data["train"]["mask_feat"].mean():.3f} ===')

    point, proba, quant, samples, imput, info = {}, {}, {}, {}, {}, {}
    t0 = time.time()
    if a.method in ('lightgbm', 'catboost'):
        fr, feat = frames_native(data, meta)
        ytr = Y['train'] if task == 'classification' else Yorig['train']          # regresi: skala standar-log
        yva = Y['val'] if task == 'classification' else Yorig['val']
        if a.method == 'lightgbm':
            m = fit_lightgbm(task, fr, ytr, yva, K, a)
        else:
            m = fit_catboost(task, fr, ytr, yva, K, a, list(meta['cat_cols']))
        info = m['info']
        for p in ('val', 'test'):
            if task == 'classification':
                proba[p] = np.asarray(m['proba'](fr[p]))
            else:
                point[p] = np.asarray(m['point'](fr[p]))
                quant[p] = np.asarray(m['quant'](fr[p]))
    else:
        mats = matrix_parts(data, meta)
        if a.method == 'grape':
            res, info = run_grape(task, mats, Y, K, a, dev)
        elif a.method == 'neumiss':
            res, info = run_neumiss(task, mats, Y, K, a, dev)
        else:
            res, imput, draws, info = run_supmiwae(task, mats, Y, K, a, dev)
            for p, dr in draws.items():
                samples[p] = inverse_y(dr, meta)
        for p in ('val', 'test'):
            if task == 'classification':
                proba[p] = res[p]
            else:
                point[p] = inverse_y(res[p], meta)
    fit_time = time.time() - t0

    summary = {'method': a.method, 'task': task, 'dataset': a.dataset, 'mech': a.mech, 'ratio': a.ratio, 'seed': a.seed,
               'classes': meta['classes'], 'info': info, 'train_time_s': fit_time, 'args': vars(a)}
    # ---------------- evaluasi ----------------
    if task == 'classification':
        T_ = mc.fit_temperature(proba['val'], Y['val'], objective='brier')
        thr = mc.fit_threshold(proba['val'], Y['val'], objective=a.thr_objective)
        summary.update({'T': T_, 'ambang': thr})
        for p in ('val', 'test'):
            m_raw = mc.cls_metrics(proba[p], Y[p], thr)
            m_ts = mc.cls_metrics(mc.apply_temperature(proba[p], T_), Y[p], thr)
            summary[p] = {'y_mentah': m_raw, 'y_ts': m_ts}
            print(f'Prediksi y ({p}) → {mc.fmt(m_raw)}{" [ambang dipilih di val ini]" if p == "val" else ""}')
    else:
        for p in ('val', 'test'):
            mm = reg_metrics(Yorig[p], point[p], samples=samples.get(p), quantiles=quant.get(p))
            summary[p] = mm
            print(f'Prediksi y ({p}) → RMSE={mm["rmse"]:.4f}  MAE={mm["mae"]:.4f}  R2={mm["r2"]:.4f}  '
                  f'CRPS={mm["crps"]:.4f}  cov min-max={mm.get("coverage_minmax", float("nan")):.3f}')
    if imput:
        for p in ('val', 'test'):
            rec = data[p]['X_obs'].copy()
            rec[:, :meta['y_idx']] = np.where(data[p]['mask'][:, :meta['y_idx']], imput[p], rec[:, :meta['y_idx']])
            mae, rmse, acc, acc_n = imputation_metrics(rec, data[p]['X_true'], data[p]['mask_feat'], meta)
            summary[p]['imputasi'] = {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n}
            print(f'Imputasi fitur ({p}) → MAE={mae:.4f}  RMSE={rmse:.4f}  ACC={acc}')

    # ---------------- simpan ----------------
    for p in ('val', 'test'):
        idx_path = f'datasets/{a.dataset}/splits/seed{a.seed}/{p}_idx.npy'
        out = pd.DataFrame({'id': load_ids(idx_path, len(data[p]['df']))})
        out['y_asli'] = data[p]['df'][meta['target']].values if task == 'classification' else Yorig[p]
        if task == 'classification':
            cl = meta['classes']
            out['y_pred'] = np.array(cl, dtype=object)[proba[p].argmax(1)]
            if summary.get('ambang') is not None:
                out['y_pred_ambang'] = np.array(cl, dtype=object)[mc.pred_at(proba[p], summary['ambang'])]
            for k, c in enumerate(cl):
                out[f'p_{c}'] = proba[p][:, k]
        else:
            out['y_pred'] = point[p]
            if p in quant:
                out['y_low'], out['y_high'] = quant[p].min(0), quant[p].max(0)
        out.to_csv(f'{pred_dir}/{name}_{p}.csv', index=False)
        if task == 'regression':          # sampel/kuantil [n, N] skala standar-log, untuk CRPS di eval_zlog.py
            S = samples.get(p, quant.get(p))
            if S is not None:
                np.save(f'{pred_dir}/{name}_{p}_samples.npy', np.asarray(S).T.astype(np.float32))
    summary['total_time_s'] = time.time() - t_all
    with open(f'{pred_dir}/{name}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    te = summary['test']
    # Klasifikasi: Acc & F1(macro) = ARGMAX (ambang 0.5), SAMA dengan metode usulan (cls_standar). F1 dengan ambang
    # dari val dicetak terpisah sebagai keterangan "[ambang val ...]" (versi lama mencetak F1 ambang sebagai F1 utama).
    ym = te.get('y_mentah', {})                                         # regresi: tidak ada y_mentah
    thr_txt = (f" [ambang val {summary['ambang']:.3f}: Acc {ym['acc_thr']:.4f} F1(macro) {ym['f1_macro_thr']:.4f}]"
               if task == 'classification' and 'f1_macro_thr' in ym else '')
    line = (f"test | {name} | " + (f"Acc {ym['acc']:.4f} F1(macro) {ym['f1_macro']:.4f} "
                                   f"AUC {ym['auc']:.4f} Brier {ym['brier']:.4f}{thr_txt}"
                                   if task == 'classification' else
                                   f"RMSE {te['rmse']:.4f} MAE {te['mae']:.4f} R2 {te['r2']:.4f} CRPS {te['crps']:.4f}")
            + f" | waktu {summary['total_time_s']:.0f}s\n")
    with open(f'{pred_dir}/result.txt', 'a') as f:
        f.write(line)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{name}_*')


if __name__ == '__main__':
    main()