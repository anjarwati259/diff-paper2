"""
Metode USULAN (modifikasi DiffPuter untuk prediksi target regresi).

File ini terpisah dari run_diffputer.py (DiffPuter asli, acuan) agar acuannya tidak berubah.
Model, sampling, metrik, dan alur EM DIPAKAI ULANG dari run_diffputer.py; yang berbeda hanya
bagian modifikasi.

Modifikasi yang sudah ada:
  [Mod 1] DWA satu arah (--dwa, aktif secara default)
          Dasar: Dynamic Weight Average, Liu dkk. CVPR 2019,
                 "End-to-End Multi-Task Learning with Attention".
          Dua kelompok: fitur dan y.
            L_k(t)  = rata-rata loss PER KOLOM kelompok k di epoch t (tanpa gradien)
            r_k     = L_k(t-1) / L_k(t-2)          (makin dekat 1 = makin lambat turun)
            w       = 2 * softmax(r / T)
            lambda_y = max(1, w_y)                  (batas bawah 1)
            bobot fitur = 1                         (dikunci)
          Dua epoch pertama tiap M-step: lambda_y = 1. Dihitung ulang dari awal di tiap iterasi EM.
          Early stopping & LR scheduler memakai loss TANPA bobot (sama dengan asli), karena loss
          berbobot berubah skala antar epoch.

Modifikasi yang menyusul: stochastic EM (Mod 3), masking (Mod 2).

Nama hasil berbeda dari asli (varian 'usulan_dwa'):
  results/mod1_dwa/{ds}/{tag}/seed{k}/usulan_dwa_test.csv, _metrics.json, _result.txt, ...
  results/mod1_dwa/{ds}/{tag}/seed{k}/usulan_dwa_lambda_iter{i}.npy   (riwayat lambda_y per epoch)
  ckpt/mod1_dwa/{ds}/{tag}/seed{k}/usulan_dwa/{trials}_{steps}/

Versi bobot lain: run_usulan_lbtw.py (LBTW satu arah) memakai alur yang sama dari file ini (fungsi run).

Contoh:
  python run_usulan.py --dataset bike --ratio 0.3 --seed 0 --quick     # uji cepat alur
  python run_usulan.py --dataset bike --ratio 0.3 --seed 0             # setting penuh
"""

import os
import copy
import json
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm

from data_loader import load_split, inverse_y, clip_bounds
from run_diffputer import (MLPDiffusion, Precond, run_trials, imputation_metrics,
                           y_metrics, save_predictions)


# ======================================================================
# Loss per kolom (EDMLoss DiffPuter, tanpa dirata-rata dulu)
# ======================================================================
def edm_loss_per_col(net, data, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """Sama dengan EDMLoss DiffPuter, tetapi mengembalikan error per elemen [B, d]."""
    sigma = (torch.randn(data.shape[0], device=data.device) * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(data) * sigma.unsqueeze(1)
    D = net(data + n, sigma)
    return weight.unsqueeze(1) * (D - data) ** 2


# ======================================================================
# [Mod 1] DWA satu arah
# ======================================================================
class OneSidedDWA:
    def __init__(self, y_idx, T=2.0):
        self.y_idx, self.T = y_idx, T
        self.hist = []                       # [(L_fitur, L_y), ...] per epoch

    def lambda_y(self):
        if len(self.hist) < 2:
            return 1.0
        (f2, y2), (f1, y1) = self.hist[-2], self.hist[-1]
        r = np.array([f1 / max(f2, 1e-12), y1 / max(y2, 1e-12)])
        w = 2 * np.exp(r / self.T) / np.exp(r / self.T).sum()
        return float(max(1.0, w[1]))

    def record(self, loss_feat_per_col, loss_y):
        self.hist.append((loss_feat_per_col, loss_y))


def train_mstep_dwa(train_data, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    torch_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    denoise_fn = MLPDiffusion(dim, args.hid_dim).to(device)
    if print_model:
        print(denoise_fn)
    net = Precond(denoise_fn).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)

    dwa = OneSidedDWA(y_idx, T=args.dwa_T)
    feat_cols = [j for j in range(dim) if j != y_idx]
    lam_hist = []

    best, patience = float('inf'), 0
    n = torch_data.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        lam = dwa.lambda_y() if args.dwa else 1.0
        col_w = torch.ones(dim, device=device)
        col_w[y_idx] = lam
        lam_hist.append(lam)

        perm = torch.randperm(n, device=device)
        tot_unw, tot_feat, tot_y = 0.0, 0.0, 0.0
        for s in range(0, n, args.batch_size):
            batch = torch_data[perm[s:s + args.batch_size]]
            err = edm_loss_per_col(net, batch)              # [B, d]
            loss = (err * col_w).mean()                     # loss berbobot untuk update
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():                           # dicatat tanpa gradien
                b = batch.shape[0]
                per_col = err.mean(0)
                tot_unw += err.mean().item() * b
                tot_feat += per_col[feat_cols].mean().item() * b
                tot_y += per_col[y_idx].item() * b

        cur = tot_unw / n                                   # loss TANPA bobot (seperti asli)
        dwa.record(tot_feat / n, tot_y / n)
        sched.step(cur)
        if cur < best:
            best, patience = cur, 0
            torch.save(net.state_dict(), f'{ckpt_iter_dir}/model.pt')
        else:
            patience += 1
            if patience == args.patience:
                print('Early stopping')
                break
        pbar.set_postfix(loss=cur, lam_y=round(lam, 3))
    pbar.close()
    net.load_state_dict(torch.load(f'{ckpt_iter_dir}/model.pt', map_location=device))
    net.eval()
    return net, np.array(lam_hist)


# ======================================================================
# E-step in-sample (dipakai run() di sini dan run_mask() di run_usulan_mask.py)
# ======================================================================
def estep_train(net, X_tr, M_tr, args, device, bounds=None):
    """E-step: isi sel yang hilang di data train.
    - DiffPuter asli (default): RATA-RATA num_trials sampel.
    - [Mod 3] Stochastic EM (args.sem = True): SATU sampel acak (1 trial), sehingga variasi
      bersyarat tetap terbawa ke iterasi berikutnya (Celeux & Diebolt 1985)."""
    if getattr(args, 'sem', False):
        a1 = copy.copy(args)
        a1.num_trials = 1
        return run_trials(net, X_tr, M_tr, a1, device, 'In-sample imputation (SEM, 1 sampel)', bounds)[0]
    return run_trials(net, X_tr, M_tr, args, device, 'In-sample imputation', bounds).mean(0)


# ======================================================================
# Alur bersama (dipakai juga oleh run_usulan_lbtw.py) — sama dengan main.py DiffPuter
# ======================================================================
def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dataset', required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mech', default='MAR')
    p.add_argument('--varian', default='auto', help="'auto' = dibuat dari modifikasi yang aktif")
    # setting DiffPuter (sama dengan run_diffputer.py)
    p.add_argument('--max_iter', type=int, default=5)
    p.add_argument('--max_epochs', type=int, default=10001)
    p.add_argument('--patience', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hid_dim', type=int, default=1024)
    p.add_argument('--num_steps', type=int, default=50)
    p.add_argument('--resample', type=int, default=20)
    p.add_argument('--num_trials', type=int, default=20)
    p.add_argument('--chunk_rows', type=int, default=50000)
    p.add_argument('--parts', nargs='+', default=['test'])
    p.add_argument('--eval_last_only', action='store_true')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--gpu', type=int, default=0)
    return p


def run(args, train_fn, info, group):
    """train_fn(X_miss, dim, y_idx, args, device, ckpt_iter_dir, print_model) -> (net, lam_hist)
    group = nama folder hasil (results/{group}/..., ckpt/{group}/...)"""
    if args.quick:
        args.max_iter, args.max_epochs, args.patience = 2, 200, 50
        args.num_steps, args.resample, args.num_trials = 10, 2, 5
        args.varian = args.varian + '_quick'

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f'{args.mech}{int(round(args.ratio * 100)):02d}'
    ckpt_dir = f'ckpt/{group}/{args.dataset}/{tag}/seed{args.seed}/{args.varian}/{args.num_trials}_{args.num_steps}'
    pred_dir = f'{args.out_dir}/{group}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    result_path = f'{pred_dir}/{args.varian}_result.txt'
    open(result_path, 'w').close()

    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech)
    dim, yi = meta['dim'], meta['y_idx']
    X_tr, M_tr = data['train']['X_obs'], data['train']['mask']
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | device {device} | '
          f'train {X_tr.shape}, val {data["val"]["X_obs"].shape}, test {data["test"]["X_obs"].shape}')
    print(f'Varian {args.varian} | {info}'
          f' | E-step: {"SEM (1 sampel)" if getattr(args, "sem", False) else "rata-rata trial (asli)"}')

    summary = {'varian': args.varian, 'dataset': args.dataset, 'mech': args.mech,
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

        # ---- M-step dengan bobot y ----
        t0 = time.time()
        net, lam_hist = train_fn(X_miss, dim, yi, args, device, f'{ckpt_dir}/{iteration}',
                                 print_model=(iteration == 0))
        train_time = time.time() - t0
        print(f'Training time: {train_time:.4f}s')
        it_res = {'iteration': iteration, 'train_time_s': train_time}
        lam_txt = ''
        if lam_hist is not None:                       # varian tanpa bobot (mis. SEM saja) -> None
            np.save(f'{pred_dir}/{args.varian}_lambda_iter{iteration}.npy', lam_hist)
            print(f'lambda_y → rata-rata {lam_hist.mean():.3f}, maks {lam_hist.max():.3f}, '
                  f'akhir {lam_hist[-1]:.3f}, epoch dengan lambda_y > 1: {(lam_hist > 1).mean() * 100:.1f}%')
            it_res['lambda_y'] = {'mean': float(lam_hist.mean()), 'max': float(lam_hist.max()),
                                  'last': float(lam_hist[-1]), 'frac_gt1': float((lam_hist > 1).mean())}
            lam_txt = f' | lambda_y mean {lam_hist.mean():.3f}'

        # ---- E-step: in-sample imputation (sama dengan asli: rata-rata trial) ----
        if M_tr.any():
            t0 = time.time()
            rec_X = estep_train(net, X_tr, M_tr, args, device, clip_bounds(meta))
            np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', rec_X)
            t_in = time.time() - t0
            print(f'In-sample imputation time: {t_in:.2f}s')
            mae, rmse, acc, acc_n = imputation_metrics(rec_X, data['train']['X_true'], data['train']['mask_feat'], meta)
            print(f'In-sample  → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}  ACC_terdekat={acc_n}')
            it_res['in_sample'] = {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n, 'time_s': t_in}
        else:
            print('Tidak ada sel hilang di train: data training tidak berubah antar iterasi')
            np.save(f'{ckpt_dir}/iter_{iteration + 1}.npy', X_tr)

        # ---- out-of-sample ----
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
                it_res[part] = {'imputasi': {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n}, 'y': ym, 'time_s': t_out}
                if last:
                    save_predictions(f'{pred_dir}/{args.varian}', part, d, S, y_samples, meta, args)
                with open(result_path, 'a') as f:
                    f.write(f'iteration {iteration} | {part} | imputasi MAE {mae:.6f} RMSE {rmse:.6f} ACC {acc} ACC_terdekat {acc_n} | '
                            f'y RMSE {ym["rmse"]:.4f} MAE {ym["mae"]:.4f} R2 {ym["r2"]:.4f} CRPS {ym["crps"]:.4f} '
                            f'cov {ym["coverage_p5_p95"]:.3f}{lam_txt}\n')

        summary['iterations'].append(it_res)
        with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

    summary['total_time_s'] = time.time() - start_all
    with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('=' * 60)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{args.varian}_*  | checkpoint: {ckpt_dir}')


# ======================================================================
# Utama: DWA satu arah
# ======================================================================
def main():
    p = build_parser('Metode usulan: DiffPuter + DWA satu arah')
    p.add_argument('--no_dwa', dest='dwa', action='store_false', help='matikan DWA (untuk ablation)')
    p.add_argument('--dwa_T', type=float, default=2.0, help='temperatur DWA (paper: 2)')
    args = p.parse_args()
    if args.varian == 'auto':
        args.varian = 'usulan_dwa' if args.dwa else 'usulan_tanpa_modifikasi'
    info = f'DWA satu arah: {"aktif (T=%g)" % args.dwa_T if args.dwa else "mati"}'
    run(args, train_mstep_dwa, info, group='mod1_dwa')


if __name__ == '__main__':
    main()