"""
PRETRAIN f(x_obs, m) untuk residual target ala CARD (Han dkk., NeurIPS 2022) — langkah 1 & 2.

f = MLP + embedding periodik PLR (Gorishniy dkk., NeurIPS 2022) yang menebak E[y | x_obs] langsung dari
data yang bolong (tanpa imputasi):
  input  : fitur numerik (PLR) + bit kategorik + indikator mask per kolom asli (1 = hilang); sel hilang = 0
           (= X_obs data_loader, skala sama dengan denoiser)
  target : y skala model data_loader x 2  (= z-score dari log1p(y) bila log_target; kode LAMA)
  loss   : MSE; AdamW; early stopping di VAL (MSE y), bobot val terbaik dipakai

Dua tahap (jalankan berurutan):
  --mode grid      seed 0: setiap konfigurasi dilatih di seluruh TRAIN, dinilai RMSE VAL. Konfigurasi terbaik
                   disimpan ke f_pretrain/{ds}/{tag}/grid_best.json (dipakai tetap untuk semua seed).
  --mode crossfit  per seed: K-fold di TRAIN dengan konfigurasi terpilih
                     - train : prediksi OUT-OF-FOLD (tiap baris dari model yang tidak melihatnya) -> residual
                               train yang jujur untuk melatih diffusion
                     - val/test : rata-rata K model fold
                   Simpan f_{train,val,test}.npy (skala model data_loader, siap jadi kolom kondisi diffusion)
                   + metrics.json. Early stopping tiap fold memakai VAL (val tidak pernah dipakai melatih).

Metrik HANYA skala standar-log (z-score dari log1p(y) dengan mean/std train = skala tempat model dilatih):
RMSE, MAE, R2 (+ disimpan di metrics.json). Nilai asli tidak dipakai. Bandingkan RMSE f dengan GBDT:
f sendiri kira-kira batas yang bisa dicapai usulan + CARD.

Contoh:
  python pretrain_f.py --mode grid --dataset bike --ratio 0.3 --seed 0
  python pretrain_f.py --mode crossfit --dataset bike --ratio 0.3 --seed 0      # lalu seed 1..4
Hasil: f_pretrain/{ds}/{tag}/grid.csv, grid_best.json, seed{k}/f_{train,val,test}.npy, seed{k}/metrics.json
"""
import os
import json
import time
import math
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import load_split
from run_usulan_plr_mask_sem_lamfix import PeriodicPLR


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def make_inputs(data, meta):
    """X: [numerik | bit kategorik] skala data_loader (hilang = 0); M: mask per kolom asli; t: target z."""
    yi = meta['y_idx']
    out = {}
    for p, d in data.items():
        X = d['X_obs'][:, :yi].astype(np.float32)
        M = np.asarray(d['mask_feat'], dtype=np.float32)
        t = (d['X_true'][:, yi] * 2.0).astype(np.float32)            # z-score skala model (log bila log_target)
        out[p] = (X, M, t)
    return out


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class FNet(nn.Module):
    """MLP-PLR + indikator mask -> 1 keluaran (E[y | x_obs])."""
    def __init__(self, n_num, n_other, n_mask, width, depth, dropout, plr_k, plr_dim, plr_sigma):
        super().__init__()
        self.n_num = n_num
        self.plr = PeriodicPLR(n_num, plr_k, plr_dim, plr_sigma) if n_num > 0 else None
        d_in = n_num * plr_dim + n_other + n_mask
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ReLU(), nn.Dropout(dropout)]
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, m):
        parts = []
        if self.plr is not None:
            parts.append(self.plr(x[:, :self.n_num]))
        parts += [x[:, self.n_num:], m]
        return self.mlp(torch.cat(parts, 1))[:, 0]


def fit_one(cfg, Xtr, Mtr, ttr, Xva, Mva, tva, meta, args, dev, seed):
    torch.manual_seed(seed)
    n_num = meta['n_num']
    net = FNet(n_num, Xtr.shape[1] - n_num, Mtr.shape[1], cfg['width'], cfg['depth'], cfg['dropout'],
               args.plr_k, args.plr_dim, args.plr_sigma).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    T = lambda a: torch.tensor(a, device=dev)
    Xtr, Mtr, ttr, Xva, Mva, tva = map(T, (Xtr, Mtr, ttr, Xva, Mva, tva))
    best, best_state, bad, ep = float('inf'), None, 0, 0
    for ep in range(args.max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for s in range(0, len(Xtr), args.batch_size):
            i = perm[s:s + args.batch_size]
            loss = ((net(Xtr[i], Mtr[i]) - ttr[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(Xva, Mva) - tva) ** 2).mean())
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state); net.eval()
    return net, math.sqrt(best), ep + 1


@torch.no_grad()
def predict(net, X, M, dev, bs=8192):
    out = []
    for s in range(0, len(X), bs):
        out.append(net(torch.tensor(X[s:s + bs], device=dev), torch.tensor(M[s:s + bs], device=dev)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------
# Metrik (skala log1p & skala asli, sama dengan cara lama)
# ----------------------------------------------------------------------
def metrics(z_pred, z_true, df_y, meta):
    """Skala standar-log saja (z-score log1p(y) statistik train), tanpa nilai asli."""
    r2 = float(1 - ((z_true - z_pred) ** 2).sum() / ((z_true - z_true.mean()) ** 2).sum())
    return {'rmse_z': float(np.sqrt(((z_pred - z_true) ** 2).mean())),
            'mae_z': float(np.abs(z_pred - z_true).mean()), 'r2_z': r2}


def fmt(m, log_t):
    return f"RMSE {m['rmse_z']:.4f}  MAE {m['mae_z']:.4f}  R2 {m['r2_z']:.4f}  (skala standar-log)"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Pretrain f(x_obs, m) untuk residual CARD')
    ap.add_argument('--mode', required=True, choices=['grid', 'crossfit'])
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mech', default='MAR')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--max_epochs', type=int, default=300)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--plr_k', type=int, default=48)
    ap.add_argument('--plr_dim', type=int, default=24)
    ap.add_argument('--plr_sigma', type=float, default=1.0)
    ap.add_argument('--widths', type=int, nargs='+', default=[256, 512])
    ap.add_argument('--depths', type=int, nargs='+', default=[3])
    ap.add_argument('--dropouts', type=float, nargs='+', default=[0.0, 0.1])
    ap.add_argument('--lrs', type=float, nargs='+', default=[1e-3, 3e-4])
    ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--out_dir', default='f_pretrain')
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    dev = f'cuda:{a.gpu}' if torch.cuda.is_available() and a.gpu >= 0 else 'cpu'

    data, meta = load_split(a.dataset, a.seed, a.ratio, a.mech)
    if meta.get('task', 'regression') != 'regression':
        raise SystemExit('pretrain_f.py hanya untuk regresi')
    I = make_inputs(data, meta)
    ydf = {p: data[p]['df'][meta['target']].to_numpy(np.float64) for p in data}
    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    root = f'{a.out_dir}/{a.dataset}/{tag}'
    os.makedirs(root, exist_ok=True)
    print(f'=== f(x_obs, m) | {a.dataset} {tag} seed{a.seed} | {a.mode} | device {dev} | log_target {meta["log_target"]} '
          f'| input {I["train"][0].shape[1]} kolom + {I["train"][1].shape[1]} mask ===', flush=True)

    if a.mode == 'grid':
        Xtr, Mtr, ttr = I['train']; Xva, Mva, tva = I['val']
        rows, best = [], None
        for w, dpt, dr, lr in itertools.product(a.widths, a.depths, a.dropouts, a.lrs):
            cfg = {'width': w, 'depth': dpt, 'dropout': dr, 'lr': lr, 'wd': a.wd}
            t0 = time.time()
            net, v, ep = fit_one(cfg, Xtr, Mtr, ttr, Xva, Mva, tva, meta, a, dev, a.seed)
            m = metrics(predict(net, Xva, Mva, dev), tva, ydf['val'], meta)
            rows.append({**cfg, 'epoch': ep, **{f'val_{k}': x for k, x in m.items()}, 'waktu_s': time.time() - t0})
            tanda = ''
            if best is None or m['rmse_z'] < best[1]:
                best, tanda = (cfg, m['rmse_z']), '  <- terbaik'
            print(f'  {cfg} ({ep} epoch, {time.time() - t0:.0f}s) val: {fmt(m, meta["log_target"])}{tanda}', flush=True)
        pd.DataFrame(rows).to_csv(f'{root}/grid.csv', index=False)
        with open(f'{root}/grid_best.json', 'w') as f:
            json.dump({'cfg': best[0], 'val_rmse_z': best[1], 'seed_grid': a.seed,
                       'plr': {'k': a.plr_k, 'dim': a.plr_dim, 'sigma': a.plr_sigma},
                       'kriteria': 'RMSE val skala standar-log'}, f, indent=2)
        print(f'Konfigurasi terpilih (val): {best[0]} -> {root}/grid_best.json')
        return

    # ---- crossfit ----
    gb = f'{root}/grid_best.json'
    if not os.path.exists(gb):
        raise SystemExit(f'{gb} belum ada. Jalankan dulu --mode grid (seed 0).')
    cfg = json.load(open(gb))['cfg']
    Xtr, Mtr, ttr = I['train']; Xva, Mva, tva = I['val']
    n = len(Xtr)
    rng = np.random.RandomState(a.seed)
    fold = rng.permutation(n) % a.folds
    oof = np.zeros(n, dtype=np.float32)
    pv, pt = [], []
    t_all = time.time()
    for k in range(a.folds):
        tr, ho = fold != k, fold == k
        net, v, ep = fit_one(cfg, Xtr[tr], Mtr[tr], ttr[tr], Xva, Mva, tva, meta, a, dev, a.seed * 100 + k)
        oof[ho] = predict(net, Xtr[ho], Mtr[ho], dev)
        pv.append(predict(net, Xva, Mva, dev)); pt.append(predict(net, *I['test'][:2], dev))
        print(f'  fold {k + 1}/{a.folds}: {ep} epoch, RMSE val (model) {v:.4f}', flush=True)
    pred = {'train': oof, 'val': np.mean(pv, 0), 'test': np.mean(pt, 0)}
    out = f'{root}/seed{a.seed}'
    os.makedirs(out, exist_ok=True)
    summary = {'cfg': cfg, 'folds': a.folds, 'waktu_s': time.time() - t_all,
               'catatan': 'f_*.npy = skala model data_loader (z/2); train = out-of-fold'}
    for p in ('train', 'val', 'test'):
        np.save(f'{out}/f_{p}.npy', (pred[p] / 2.0).astype(np.float32))
        m = metrics(pred[p], I[p][2], ydf[p], meta)
        summary[p] = m
        print(f'f saja ({p}{", out-of-fold" if p == "train" else ""}) → {fmt(m, meta["log_target"])}', flush=True)
    r = I['train'][2] - oof
    summary['residual_train_oof'] = {'mean': float(r.mean()), 'std': float(r.std())}
    print(f'Residual train (out-of-fold, skala z): mean {r.mean():.4f}  std {r.std():.4f}  '
          f'(dipakai menskala ulang residual untuk diffusion)')
    with open(f'{out}/metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Selesai ({summary["waktu_s"]:.0f}s). Hasil: {out}/')


if __name__ == '__main__':
    main()