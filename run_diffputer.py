"""
Langkah 2: DiffPuter-prediksi ASLI (acuan untuk semua perbandingan).

Algoritma sama dengan kode resmi DiffPuter (model.py, diffusion_utils.py, main.py):
  - MLPDiffusion + Precond (EDM), EDMLoss (noise & loss di semua kolom, bobot sama)
  - impute_mask: Heun + churn, resampling 20x per langkah, sel teramati = asli + noise
  - Loop EM: inisialisasi 0 (= mean), M-step latih model baru, E-step isi ulang dengan
    RATA-RATA N sampel; early stopping dari loss training (patience 500)

Alur dan tampilan output mengikuti main.py DiffPuter, per iterasi EM:
  Training (M-step) -> In-sample imputation (E-step di train, loop num_trials)
  -> Out-of-sample imputation (test; val juga bila diminta) -> tulis result.txt

Yang berubah (bukan algoritma):
  - Data dari data_loader.py (split/mask per seed; y kolom terakhir)
  - Di val/test, kolom y ikut ditebak bersama fitur yang hilang
  - Selain MAE/RMSE/ACC imputasi, dicetak juga metrik prediksi y (skala asli)
  - Output akhir: CSV prediksi, sampel y (.npy), metrik (.json)
    di folder results/diffputer_asli/{ds}/{tag}/seed{k}/ (checkpoint: ckpt/diffputer_asli/...)

Modifikasi usulan (bobot, masking, stochastic EM) BELUM ada di file ini;
nanti ditambahkan sebagai pilihan --varian.

Contoh:
  python run_diffputer.py --dataset bike --ratio 0.3 --seed 0 --quick          # uji cepat alur
  python run_diffputer.py --dataset bike --ratio 0.3 --seed 0                  # setting penuh
  python run_diffputer.py --dataset bike --ratio 0.3 --seed 0 --num_trials 10  # sama seperti run lamamu (10_50)
  python run_diffputer.py ... --parts val test                                 # evaluasi val juga (untuk tuning)
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import load_split, inverse_transform, inverse_y, clip_bounds, model_to_eval

# ======================================================================
# Model (salinan model.py DiffPuter)
# ======================================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, noise_labels):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)
        return self.mlp(self.proj(x) + emb)


class Precond(nn.Module):
    def __init__(self, denoise_fn, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
        super().__init__()
        self.sigma_min, self.sigma_max, self.sigma_data = sigma_min, sigma_max, sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.denoise_fn_F(c_in * x, c_noise.flatten())
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def edm_loss(net, data, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """EDMLoss DiffPuter: satu level noise per baris, noise & loss di semua kolom (bobot sama)."""
    sigma = (torch.randn(data.shape[0], device=data.device) * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(data) * sigma.unsqueeze(1)
    D = net(data + n, sigma)
    return (weight.unsqueeze(1) * (D - data) ** 2).mean()


# ======================================================================
# Sampling (salinan diffusion_utils.py DiffPuter, ditambah pemrosesan per batch)
# ======================================================================
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80, 7
S_CHURN, S_MIN, S_MAX, S_NOISE = 1, 0, float('inf'), 1


def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next
    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


@torch.no_grad()
def impute_mask(net, x, mask, num_steps=50, resample=20):
    """x: nilai diketahui (sel tersembunyi = 0); mask: 1 = ditebak. Sama dengan DiffPuter."""
    device = x.device
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min, sigma_max = max(SIGMA_MIN, net.sigma_min), min(SIGMA_MAX, net.sigma_max)
    t_steps = (sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    mask = mask.to(torch.float32)
    x_t = torch.randn_like(x) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        reps = resample if i < num_steps - 1 else 1
        for j in range(reps):
            x_known = x + torch.randn_like(x) * t_next
            x_unknown = sample_step(net, num_steps, i, t_cur, t_next, x_t)
            x_t_prev = x_known * (1 - mask) + x_unknown * mask
            if j == reps - 1:
                x_t = x_t_prev
            else:
                x_t = x_t_prev + torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()
    return x_t


@torch.no_grad()
def impute_once(net, X_obs, mask, num_steps, resample, chunk_rows, device):
    """Satu trial imputasi untuk semua baris (diproses per potongan bila datanya besar).
    Sel yang diketahui tetap nilai asli (rec_X * mask + impute_X * (1 - mask))."""
    out = np.empty_like(X_obs, dtype=np.float32)
    for s in range(0, X_obs.shape[0], chunk_rows):
        xb = torch.tensor(X_obs[s:s + chunk_rows], dtype=torch.float32, device=device)
        mb = torch.tensor(mask[s:s + chunk_rows], dtype=torch.float32, device=device)
        rec = impute_mask(net, xb, mb, num_steps, resample)
        out[s:s + chunk_rows] = (rec * mb + xb * (1 - mb)).cpu().numpy()
    return out


def clip_generated(R, mask, bounds, desc=''):
    """Batasi nilai HASIL GENERATE (sel mask = 1) ke rentang data train +- margin (skala terstandardisasi).
    Sel yang diketahui tidak disentuh. Mencegah satu lintasan sampling yang lepas kendali merusak
    rata-rata (dan expm1 pada target log). Mencetak jumlah sel yang terpotong."""
    if bounds is None:
        return R
    lo, hi = bounds
    m = np.broadcast_to(np.asarray(mask, dtype=bool), R.shape)
    C = np.clip(R, lo, hi)
    k = int(((C != R) & m).sum())
    if k:
        print(f'[clip] {desc}: {k} sel hasil generate dipotong ke rentang train '
              f'({k / max(int(m.sum()), 1) * 100:.4f}% sel yang ditebak)')
    return np.where(m, C, R)


def run_trials(net, X_obs, mask, args, device, desc, bounds=None):
    """Loop num_trials seperti main.py DiffPuter. Kembali: array [num_trials, n, d].
    bounds = (lo, hi) dari data_loader.clip_bounds(meta) -> hasil generate di-clip."""
    recs = []
    for _ in tqdm(range(args.num_trials), desc=desc):
        recs.append(impute_once(net, X_obs, mask, args.num_steps, args.resample, args.chunk_rows, device))
    return clip_generated(np.stack(recs, axis=0), mask, bounds, desc)


# ======================================================================
# M-step (training) — sama dengan main.py DiffPuter
# ======================================================================
def train_mstep(train_data, dim, args, device, ckpt_iter_dir, print_model):
    torch_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    denoise_fn = MLPDiffusion(dim, args.hid_dim).to(device)
    if print_model:
        print(denoise_fn)
    net = Precond(denoise_fn).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)

    best, patience = float('inf'), 0
    n = torch_data.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, args.batch_size):
            batch = torch_data[perm[s:s + args.batch_size]]
            loss = edm_loss(net, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * batch.shape[0]
        cur = tot / n
        sched.step(cur)
        if cur < best:
            best, patience = cur, 0
            torch.save(net.state_dict(), f'{ckpt_iter_dir}/model.pt')
        else:
            patience += 1
            if patience == args.patience:
                print('Early stopping')
                break
        pbar.set_postfix(loss=cur)
    pbar.close()
    net.load_state_dict(torch.load(f'{ckpt_iter_dir}/model.pt', map_location=device))
    net.eval()
    return net


# ======================================================================
# Metrik
# ======================================================================
def imputation_metrics(rec_mean, X_true, mask_feat, meta):
    """Seperti get_eval DiffPuter: MAE/RMSE fitur numerik di sel yang di-mask
    (skala terstandardisasi, X*2 seperti main.py) dan ACC fitur kategorik.
    Hanya sel FITUR yang missing yang dihitung (kolom y tidak ikut).

    Dua versi ACC:
      acc          : KETAT, aturan repo DiffPuter (recover_num_cat di baselines): tiap bit dibulatkan
                     (> 0.5 -> 1), dibaca sebagai biner; kode yang tidak dipakai kategori mana pun
                     dihitung SALAH.  -> angka utama
      acc_terdekat : kategori valid dengan kode biner terdekat (decode_cats).  -> catatan tambahan
    Kembali: (mae, rmse, acc, acc_terdekat)
    rec_mean di skala MODEL; bila ada transformasi fitur numerik (meta['tf']) dikembalikan dulu ke skala asli
    (model_to_eval). X_true selalu di skala asli. Tanpa transformasi: tidak ada perubahan."""
    rec_mean = model_to_eval(rec_mean, meta)
    n_num = meta['n_num']
    m_num = mask_feat[:, :n_num]
    if m_num.any():
        diff = (rec_mean[:, :n_num] - X_true[:, :n_num])[m_num] * 2
        mae, rmse = float(np.abs(diff).mean()), float(np.sqrt((diff ** 2).mean()))
    else:
        mae = rmse = float('nan')
    acc = acc_n = float('nan')
    if meta['cat_cols']:
        m_cat = mask_feat[:, n_num:]
        if m_cat.any():
            true = inverse_transform(X_true, meta)
            # versi terdekat
            pred = inverse_transform(rec_mean, meta)
            hits_n = [(pred[c].astype(str).values == true[c].astype(str).values)[m_cat[:, j]]
                      for j, c in enumerate(meta['cat_cols'])]
            acc_n = float(np.concatenate(hits_n).mean())
            # versi ketat (repo DiffPuter)
            bits = (rec_mean * 2.0 * meta['std'] + meta['mean'])[:, n_num:meta['y_idx']]
            hits, start = [], 0
            for j, c in enumerate(meta['cat_cols']):
                cm = meta['cat_maps'][c]
                b = (bits[:, start:start + cm['bits']] > 0.5).astype(np.int64)
                code = (b * (2 ** np.arange(cm['bits'] - 1, -1, -1))).sum(1)   # bit pertama = paling kiri
                idx_true = {v: i for i, v in enumerate(cm['categories'])}
                t = np.array([idx_true.get(v, -1) for v in true[c].tolist()])
                hits.append((code == t)[m_cat[:, j]])                         # kode tak valid -> salah
                start += cm['bits']
            acc = float(np.concatenate(hits).mean())
    return mae, rmse, acc, acc_n


def crps_fair(samples, y):
    """samples [N, n], y [n] -> rata-rata CRPS (versi fair/tak bias)."""
    N = samples.shape[0]
    t1 = np.abs(samples - y[None, :]).mean(0)
    t2 = np.abs(samples[:, None, :] - samples[None, :, :]).sum((0, 1)) / (2 * N * (N - 1))
    return float((t1 - t2).mean())


def y_metrics(y_true, samples):
    y_pred = samples.mean(0)
    lo, hi = np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)
    mn, mx = samples.min(0), samples.max(0)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    rank = (samples < y_true[None, :]).sum(0)                 # 0..N
    hist = np.bincount(rank, minlength=samples.shape[0] + 1) / len(y_true)
    return {
        'rmse': float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        'mae': float(np.abs(y_true - y_pred).mean()),
        'r2': float(1 - ss_res / ss_tot),
        'crps': crps_fair(samples, y_true),
        'coverage_p5_p95': float(((y_true >= lo) & (y_true <= hi)).mean()),
        'width_p5_p95': float((hi - lo).mean()),
        'coverage_minmax': float(((y_true >= mn) & (y_true <= mx)).mean()),
        'coverage_minmax_ideal': (samples.shape[0] - 1) / (samples.shape[0] + 1),
        'width_minmax': float((mx - mn).mean()),
        'rank_hist_dev': float(np.abs(hist - 1 / len(hist)).mean()),
    }


def save_predictions(path_prefix, part, d, S, y_samples, meta, args):
    """CSV: id, fitur asli (NaN jika di-mask), {fitur}_isian, y_asli, y_pred, y_low, y_high."""
    df = d['df']
    feat_cols = meta['num_cols'] + meta['cat_cols']
    out = pd.DataFrame({'id': np.load(f"datasets/{args.dataset}/splits/seed{args.seed}/{part}_idx.npy")})
    shown = df[feat_cols].mask(d['mask_feat'])
    imputed = inverse_transform(model_to_eval(S.mean(0), meta), meta)
    for j, c in enumerate(feat_cols):
        out[c] = shown[c].values
        if d['mask_feat'][:, j].any():
            out[f'{c}_isian'] = np.where(d['mask_feat'][:, j], imputed[c].values, None)
    out['y_asli'] = df[meta['target']].to_numpy(dtype=np.float64)
    out['y_pred'] = y_samples.mean(0)
    out['y_low'] = np.percentile(y_samples, 5, axis=0)
    out['y_high'] = np.percentile(y_samples, 95, axis=0)
    out.to_csv(f'{path_prefix}_{part}.csv', index=False)
    np.save(f'{path_prefix}_{part}_samples.npy', y_samples.T.astype(np.float32))


# ======================================================================
# Utama — alur mengikuti main.py DiffPuter
# ======================================================================
def main():
    p = argparse.ArgumentParser(description='DiffPuter-prediksi asli')
    p.add_argument('--dataset', required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mech', default='MAR')
    p.add_argument('--varian', default='diffputer_asli')
    p.add_argument('--max_iter', type=int, default=5)         # K (iterasi EM)
    p.add_argument('--max_epochs', type=int, default=10001)   # sama dengan main.py
    p.add_argument('--patience', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hid_dim', type=int, default=1024)
    p.add_argument('--num_steps', type=int, default=50)       # M
    p.add_argument('--resample', type=int, default=20)        # loop dalam impute_mask
    p.add_argument('--num_trials', type=int, default=20)      # N sampel
    p.add_argument('--chunk_rows', type=int, default=50000)   # kecilkan jika memori GPU kurang
    p.add_argument('--parts', nargs='+', default=['test'], help="'test' (seperti asli) atau 'val test'")
    p.add_argument('--eval_last_only', action='store_true',
                   help='out-of-sample hanya di iterasi terakhir (lebih cepat; asli = tiap iterasi)')
    p.add_argument('--quick', action='store_true', help='uji cepat alur (hasil TIDAK untuk dilaporkan)')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    if args.quick:
        args.max_iter, args.max_epochs, args.patience = 2, 200, 50
        args.num_steps, args.resample, args.num_trials = 10, 2, 5
        args.varian = args.varian + '_quick'

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f'{args.mech}{int(round(args.ratio * 100)):02d}'
    ckpt_dir = f'ckpt/diffputer_asli/{args.dataset}/{tag}/seed{args.seed}/{args.varian}/{args.num_trials}_{args.num_steps}'
    pred_dir = f'{args.out_dir}/diffputer_asli/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    result_path = f'{pred_dir}/{args.varian}_result.txt'
    open(result_path, 'w').close()

    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech)
    dim, yi = meta['dim'], meta['y_idx']
    X_tr, M_tr = data['train']['X_obs'], data['train']['mask']
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | device {device} | '
          f'train {X_tr.shape}, val {data["val"]["X_obs"].shape}, test {data["test"]["X_obs"].shape}')

    summary = {'varian': args.varian, 'dataset': args.dataset, 'mech': args.mech,
               'ratio': args.ratio, 'seed': args.seed, 'args': vars(args), 'iterations': []}
    start_all = time.time()

    for iteration in range(args.max_iter):
        print('=' * 60)
        print(f'Iteration: {iteration}')
        print(f'Checkpoint dir: {ckpt_dir}')
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        # ---- data training iterasi ini ----
        if iteration == 0:
            X_miss = X_tr.copy()                           # sel hilang = 0 (= mean)
        else:
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            X_miss = np.load(f'{ckpt_dir}/iter_{iteration}.npy')
        print(f'[INFO] X_miss shape: {X_miss.shape}, range: [{X_miss.min():.4f}, {X_miss.max():.4f}]')

        # ---- M-step ----
        t0 = time.time()
        net = train_mstep(X_miss, dim, args, device, f'{ckpt_dir}/{iteration}', print_model=(iteration == 0))
        train_time = time.time() - t0
        print(f'Training time: {train_time:.4f}s')
        it_res = {'iteration': iteration, 'train_time_s': train_time}

        # ---- E-step: in-sample imputation (train) ----
        if M_tr.any():
            t0 = time.time()
            R = run_trials(net, X_tr, M_tr, args, device, 'In-sample imputation', clip_bounds(meta))
            rec_X = R.mean(0)                              # DiffPuter asli: rata-rata trial
            np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', rec_X)
            t_in = time.time() - t0
            print(f'In-sample imputation time: {t_in:.2f}s')
            mae, rmse, acc, acc_n = imputation_metrics(rec_X, data['train']['X_true'], data['train']['mask_feat'], meta)
            print(f'In-sample  → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}  ACC_terdekat={acc_n}')
            it_res['in_sample'] = {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n, 'time_s': t_in}
        else:
            print('Tidak ada sel hilang di train: data training tidak berubah antar iterasi')
            np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', X_tr)

        # ---- out-of-sample (test, dan val bila diminta) ----
        last = iteration == args.max_iter - 1
        if last or not args.eval_last_only:
            for part in args.parts:
                d = data[part]
                t0 = time.time()
                S = run_trials(net, d['X_obs'], d['mask'], args, device, f'Out-of-sample imputation ({part})',
                               clip_bounds(meta))
                t_out = time.time() - t0
                print(f'Out-of-sample imputation time ({part}): {t_out:.2f}s')
                mae, rmse, acc, acc_n = imputation_metrics(S.mean(0), d['X_true'], d['mask_feat'], meta)
                y_samples = inverse_y(S[:, :, yi], meta)
                np.save(f'{pred_dir}/{args.varian}_{part}_samples_iter{iteration}.npy',
                        y_samples.T.astype(np.float32))          # sampel y per iterasi (diagnosa)
                ym = y_metrics(d['df'][meta['target']].to_numpy(dtype=np.float64), y_samples)
                print(f'Out-of-sample ({part}) → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}  ACC_terdekat={acc_n}')
                print(f'Prediksi y ({part}) → RMSE={ym["rmse"]:.4f}  MAE={ym["mae"]:.4f}  R2={ym["r2"]:.4f}  '
                      f'CRPS={ym["crps"]:.4f}  cov p5-95={ym["coverage_p5_p95"]:.3f}  '
                      f'cov min-max={ym["coverage_minmax"]:.3f} (ideal {ym["coverage_minmax_ideal"]:.3f})')
                it_res[part] = {'imputasi': {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n},
                                'y': ym, 'time_s': t_out}
                if last:
                    save_predictions(f'{pred_dir}/{args.varian}', part, d, S, y_samples, meta, args)

                with open(result_path, 'a') as f:
                    f.write(f'iteration {iteration} | {part} | imputasi MAE {mae:.6f} RMSE {rmse:.6f} ACC {acc} ACC_terdekat {acc_n} | '
                            f'y RMSE {ym["rmse"]:.4f} MAE {ym["mae"]:.4f} R2 {ym["r2"]:.4f} CRPS {ym["crps"]:.4f} '
                            f'cov {ym["coverage_p5_p95"]:.3f}\n')

        summary['iterations'].append(it_res)
        with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

    summary['total_time_s'] = time.time() - start_all
    with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('=' * 60)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{args.varian}_*  | checkpoint: {ckpt_dir}')


if __name__ == '__main__':
    main()