"""
Metode USULAN — Modifikasi 2: target-aware masking (TANPA bobot).

Ide: DiffPuter asli mengkondisikan secara tidak langsung (fitur teramati diberi noise lalu ditimpa
tiap langkah sampling). Di sini model juga dilatih menebak y (dan fitur yang bolong) dengan fitur
teramati diberikan BERSIH, plus penanda c (1 = ditebak/diberi noise, 0 = bersih/diketahui).

Training (M-step), tiap batch, tiap baris dilempar koin:
  - peluang 1-p : baris FULL NOISE  -> c = 1 di semua kolom (= DiffPuter asli)
  - peluang p   : baris KONDISIONAL -> c = 1 hanya di y dan fitur yang aslinya bolong
                                       (isian E-step), fitur teramati bersih (c = 0)
  Noise (sigma sama per baris) hanya ditambahkan ke sel c = 1. Loss hanya di sel c = 1.
Jaringan: input = [x_in, c] (dimensi 2d), output d. Precond: x_in = c_in*x*c + x*(1-c)
  (sel bersih tidak diskalakan c_in).

E-step (in-sample, isi data train): TETAP cara asli DiffPuter (mode full noise, c = 1 semua,
  sel teramati ditimpa tiap langkah, resample 20x, rata-rata trial).
Test (out-of-sample): MODE KONDISIONAL — fitur teramati bersih (c = 0) dan tetap; fitur bolong dan
  y ditebak dari noise (c = 1). --resample_test mengatur loop resample (default sama dengan asli: 20).

Selain itu (loop EM, K, epoch, early stopping, jumlah trial, metrik) sama dengan run_diffputer.py.
Early stopping & scheduler memakai loss rata-rata di sel c = 1.

Hasil di folder TERPISAH:
  results/mod2_masking/{ds}/{tag}/seed{k}/usulan_mask_p{p}_*
  ckpt/mod2_masking/{ds}/{tag}/seed{k}/usulan_mask_p{p}/{trials}_{steps}/
(Versi gabungan dengan bobot LBTW: run_usulan_mask_lbtw.py -> results/mod2_masking_lbtw/...)

Contoh:
  python run_usulan_mask.py --dataset bike --ratio 0.3 --seed 0 --quick
  python run_usulan_mask.py --dataset bike --ratio 0.3 --seed 0                 # p = 0.5
  python run_usulan_mask.py --dataset bike --ratio 0.3 --seed 0 --resample_test 1
"""

import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import load_split, inverse_y, clip_bounds
from run_diffputer import (PositionalEmbedding, run_trials, imputation_metrics, y_metrics, clip_generated,
                           save_predictions, SIGMA_MIN, SIGMA_MAX, RHO, S_CHURN, S_MIN, S_MAX, S_NOISE)
from run_usulan import build_parser, estep_train


# ======================================================================
# Model: MLPDiffusion DiffPuter + input penanda c
# ======================================================================
class MLPDiffusionMask(nn.Module):
    """Sama dengan MLPDiffusion DiffPuter, kecuali lapisan proj menerima [x, c] (2d)."""
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.proj = nn.Linear(2 * d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, c, noise_labels):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        return self.mlp(self.proj(torch.cat([x, c], dim=1)) + emb)


class PrecondMask(nn.Module):
    """Precond EDM DiffPuter + penanda c. c=None -> semua 1 (mode full noise = asli),
    sehingga impute_mask/run_trials asli bisa dipakai apa adanya untuk E-step."""
    def __init__(self, denoise_fn, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5):
        super().__init__()
        self.sigma_min, self.sigma_max, self.sigma_data = sigma_min, sigma_max, sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, c=None):
        x = x.to(torch.float32)
        if c is None:
            c = torch.ones_like(x)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        if sigma.shape[0] == 1 and x.shape[0] > 1:
            sigma = sigma.expand(x.shape[0], 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        x_in = c_in * x * c + x * (1 - c)                 # sel bersih tidak diskalakan
        F_x = self.denoise_fn_F(x_in, c, c_noise.flatten())
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def build_net(dim, args, device, print_model):
    denoise_fn = MLPDiffusionMask(dim, args.hid_dim).to(device)
    if print_model:
        print(denoise_fn)
    return PrecondMask(denoise_fn).to(device)


# ======================================================================
# Penanda c dan loss
# ======================================================================
def make_cond(mask_rows, p, y_idx):
    """mask_rows: [B, d] 1 = sel aslinya bolong (dari mask train). Kembali c [B, d]:
    baris kondisional (peluang p): c = bolong + y ; baris full noise: c = 1 semua."""
    B, d = mask_rows.shape
    c_cond = mask_rows.clone()
    c_cond[:, y_idx] = 1.0
    is_cond = (torch.rand(B, 1, device=mask_rows.device) < p).float()
    return is_cond * c_cond + (1 - is_cond) * torch.ones_like(c_cond)


def edm_loss_mask(net, data, c, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """EDMLoss DiffPuter dengan penanda: noise & loss hanya di sel c = 1. Kembali error [B, d]
    (sel c = 0 bernilai 0)."""
    sigma = (torch.randn(data.shape[0], device=data.device) * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(data) * sigma.unsqueeze(1) * c
    D = net(data + n, sigma, c)
    return weight.unsqueeze(1) * (D - data) ** 2 * c


def train_mstep_mask(train_data, train_mask, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    """M-step masking tanpa bobot. Kembali (net, None)."""
    X = torch.tensor(train_data, dtype=torch.float32, device=device)
    M = torch.tensor(train_mask, dtype=torch.float32, device=device)
    net = build_net(dim, args, device, print_model)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)

    best, patience = float('inf'), 0
    n = X.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        perm = torch.randperm(n, device=device)
        tot_err, tot_cnt = 0.0, 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            c = make_cond(M[idx], args.p_cond, y_idx)
            err = edm_loss_mask(net, X[idx], c)
            loss = err.sum() / c.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_err += err.sum().item()
            tot_cnt += c.sum().item()
        cur = tot_err / tot_cnt
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
    return net, None


# ======================================================================
# Sampling kondisional (test): Heun + churn DiffPuter, sel c = 0 bersih & tetap
# ======================================================================
def sample_step_cond(net, c, x_clean, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next
    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1) if S_MIN <= t_cur <= S_MAX else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur) * c
    denoised = net(x_hat, t_hat, c).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    if i < num_steps - 1:
        denoised = net(x_next, t_next, c).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_clean * (1 - c) + x_next * c                 # sel diketahui tetap bersih


@torch.no_grad()
def impute_cond(net, x, mask, num_steps=50, resample=20):
    """x: nilai diketahui (sel tersembunyi = 0); mask: 1 = ditebak (= c)."""
    device = x.device
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min, sigma_max = max(SIGMA_MIN, net.sigma_min), min(SIGMA_MAX, net.sigma_max)
    t_steps = (sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    c = mask.to(torch.float32)
    x_t = x * (1 - c) + torch.randn_like(x) * t_steps[0] * c
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        reps = resample if i < num_steps - 1 else 1
        for j in range(reps):
            x_new = sample_step_cond(net, c, x, num_steps, i, t_cur, t_next, x_t)
            if j == reps - 1:
                x_t = x_new
            else:                                          # kembali ke level t_cur (seperti asli)
                x_t = x_new + torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt() * c
    return x_t


@torch.no_grad()
def run_trials_cond(net, X_obs, mask, args, device, desc, bounds=None):
    recs = []
    for _ in tqdm(range(args.num_trials), desc=desc):
        out = np.empty_like(X_obs, dtype=np.float32)
        for s in range(0, X_obs.shape[0], args.chunk_rows):
            xb = torch.tensor(X_obs[s:s + args.chunk_rows], dtype=torch.float32, device=device)
            mb = torch.tensor(mask[s:s + args.chunk_rows], dtype=torch.float32, device=device)
            rec = impute_cond(net, xb, mb, args.num_steps, args.resample_test)
            out[s:s + args.chunk_rows] = (rec * mb + xb * (1 - mb)).cpu().numpy()
        recs.append(out)
    return clip_generated(np.stack(recs, axis=0), mask, bounds, desc)


# ======================================================================
# Alur EM (sama dengan run() di run_usulan.py; beda: mask train dikirim ke M-step,
# test memakai sampling kondisional, folder hasil terpisah per kelompok)
# ======================================================================
def add_mask_args(p):
    p.add_argument('--p_cond', type=float, default=0.5, help='peluang baris kondisional saat training')
    p.add_argument('--resample_test', type=int, default=None,
                   help='loop resample saat test (default = --resample, yaitu 20 seperti asli)')
    return p


def run_mask(args, train_fn, info, group):
    if args.quick:
        args.max_iter, args.max_epochs, args.patience = 2, 200, 50
        args.num_steps, args.resample, args.num_trials = 10, 2, 5
        args.varian = args.varian + '_quick'
    if args.resample_test is None:
        args.resample_test = args.resample

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f'{args.mech}{int(round(args.ratio * 100)):02d}'
    ckpt_dir = f'ckpt/{group}/{args.dataset}/{tag}/seed{args.seed}/{args.varian}/{args.num_trials}_{args.num_steps}'
    pred_dir = f'{args.out_dir}/{group}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    result_path = f'{pred_dir}/{args.varian}_result.txt'
    open(result_path, 'w').close()

    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech,
                            num_transform=getattr(args, 'num_transform', 'none'))
    dim, yi = meta['dim'], meta['y_idx']
    X_tr, M_tr = data['train']['X_obs'], data['train']['mask']
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | device {device} | '
          f'train {X_tr.shape}, val {data["val"]["X_obs"].shape}, test {data["test"]["X_obs"].shape}')
    print(f'Varian {args.varian} | masking p={args.p_cond:g}, resample test={args.resample_test} | {info}'
          f' | E-step: {"SEM (1 sampel)" if getattr(args, "sem", False) else "rata-rata trial (asli)"}')

    summary = {'varian': args.varian, 'kelompok': group, 'dataset': args.dataset, 'mech': args.mech,
               'ratio': args.ratio, 'seed': args.seed, 'args': vars(args), 'iterations': []}
    start_all = time.time()

    for iteration in range(args.max_iter):
        print('=' * 60)
        print(f'Iteration: {iteration}')
        print(f'Checkpoint dir: {ckpt_dir}')
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        if iteration == 0:
            X_miss = X_tr.copy()
        else:
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            X_miss = np.load(f'{ckpt_dir}/iter_{iteration}.npy')
        print(f'[INFO] X_miss shape: {X_miss.shape}, range: [{X_miss.min():.4f}, {X_miss.max():.4f}]')

        # ---- M-step (masking) ----
        t0 = time.time()
        net, lam_hist = train_fn(X_miss, M_tr, dim, yi, args, device, f'{ckpt_dir}/{iteration}',
                                 print_model=(iteration == 0))
        train_time = time.time() - t0
        print(f'Training time: {train_time:.4f}s')
        it_res = {'iteration': iteration, 'train_time_s': train_time}
        lam_txt = ''
        if lam_hist is not None:
            np.save(f'{pred_dir}/{args.varian}_lambda_iter{iteration}.npy', lam_hist)
            print(f'lambda_y → rata-rata {lam_hist.mean():.3f}, maks {lam_hist.max():.3f}, '
                  f'akhir {lam_hist[-1]:.3f}, epoch dengan lambda_y > 1: {(lam_hist > 1).mean() * 100:.1f}%')
            it_res['lambda_y'] = {'mean': float(lam_hist.mean()), 'max': float(lam_hist.max()),
                                  'last': float(lam_hist[-1]), 'frac_gt1': float((lam_hist > 1).mean())}
            lam_txt = f' | lambda_y mean {lam_hist.mean():.3f}'

        # ---- E-step: in-sample imputation, CARA ASLI (mode full noise) ----
        if M_tr.any():
            t0 = time.time()
            rec_X = estep_train(net, X_tr, M_tr, args, device, clip_bounds(meta))   # rata-rata (asli) atau SEM
            np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', rec_X)
            t_in = time.time() - t0
            print(f'In-sample imputation time: {t_in:.2f}s')
            mae, rmse, acc, acc_n = imputation_metrics(rec_X, data['train']['X_true'], data['train']['mask_feat'], meta)
            print(f'In-sample  → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}  ACC_terdekat={acc_n}')
            it_res['in_sample'] = {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n, 'time_s': t_in}
        else:
            print('Tidak ada sel hilang di train: data training tidak berubah antar iterasi')
            np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', X_tr)

        # ---- out-of-sample: MODE KONDISIONAL ----
        last = iteration == args.max_iter - 1
        if last or not args.eval_last_only:
            for part in args.parts:
                d = data[part]
                t0 = time.time()
                S = run_trials_cond(net, d['X_obs'], d['mask'], args, device,
                                    f'Out-of-sample imputation kondisional ({part})', clip_bounds(meta))
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
                it_res[part] = {'imputasi': {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n}, 'y': ym, 'time_s': t_out}
                if last:
                    save_predictions(f'{pred_dir}/{args.varian}', part, d, S, y_samples, meta, args)
                with open(result_path, 'a') as f:
                    f.write(f'iteration {iteration} | {part} | imputasi MAE {mae:.6f} RMSE {rmse:.6f} ACC {acc} ACC_terdekat {acc_n} | '
                            f'y RMSE {ym["rmse"]:.4f} MAE {ym["mae"]:.4f} R2 {ym["r2"]:.4f} CRPS {ym["crps"]:.4f} '
                            f'cov {ym["coverage_p5_p95"]:.3f}{lam_txt} | waktu test {t_out:.0f}s\n')

        summary['iterations'].append(it_res)
        with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

    summary['total_time_s'] = time.time() - start_all
    with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('=' * 60)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{args.varian}_*  | checkpoint: {ckpt_dir}')


def varian_suffix(args):
    s = f'p{args.p_cond:g}'
    if args.resample_test is not None and args.resample_test != args.resample:
        s += f'_rt{args.resample_test}'
    return s


def main():
    p = add_mask_args(build_parser('Metode usulan: DiffPuter + target-aware masking (Mod 2)'))
    args = p.parse_args()
    if args.varian == 'auto':
        args.varian = f'usulan_mask_{varian_suffix(args)}'
    run_mask(args, train_mstep_mask, 'tanpa bobot', group='mod2_masking')


if __name__ == '__main__':
    main()