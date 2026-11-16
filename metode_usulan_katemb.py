"""
METODE USULAN + SAKLAR EMBEDDING KATEGORIK (--emb_kategorik). Salinan metode_usulan.py; satu-satunya tambahan
adalah saklar ini. Hasil di folder TERPISAH: results_zlog/usulan_katemb/... (tidak tercampur).

--emb_kategorik (default MATI = identik dengan metode_usulan.py), VERSI SEDERHANA, hanya sisi INPUT jaringan:
    Setiap fitur kategorik j punya tabel embedding E_j [jumlah kategori x d] (d = --plr_dim, default 24),
    sejajar dengan PLR untuk fitur numerik (FT-Transformer, Gorishniy dkk. 2021; entity embeddings).
      sel BERSIH (penanda c = 0, kategori teramati): bit di-decode ke indeks kategori -> E_j[k]  (lookup)
      sel BER-NOISE (c = 1, kategori hilang/ditebak): bit mentah masuk apa adanya (seperti sekarang)
    Input ke proj per fitur kategorik = [E_j[k] * (1 - c), bit * c]. PLR, MLP, OUTPUT (bit), loss, sampler,
    E-step, decode kategori, metrik: TIDAK berubah. Butuh PLR dan masking nyala. Nama varian + '_katemb'.
    Contoh: python metode_usulan_katemb.py --dataset shoppers --ratio 0.3 --seed 2 --parts val test --emb_kategorik --hapus_ckpt
    Rekap : python rekap_usulan.py --root results_zlog/usulan_katemb

------------------------------------------------------------------------------------------------------------
METODE USULAN — satu skrip untuk REGRESI dan KLASIFIKASI, komponen bisa dibongkar pasang (ablasi).

Metode final (DEFAULT, tanpa saklar apa pun):
    target-aware masking (p = 0.5) + bobot y lambda = 10 + PLR (K = 48, d = 24, sigma = 1)
    E-step cara DiffPuter (rata-rata N = 10 sampel, mode full noise), 5 iterasi EM
    prediksi = posterior mean, model iterasi EM terakhir (langkah sigma dipilih di VAL)
    klasifikasi: y one-hot +-0.5, loss MSE (opsi --loss_y ce)

Saklar ablasi (nyala/mati):
    --masking / --no_masking   target-aware masking + penanda c. Mati = semua kolom diberi noise dan sampling uji
                               full noise + penimpaan (cara DiffPuter).
    --plr / --no_plr           embedding PLR untuk kolom numerik (regresi: + y).
    --lam X                    bobot TOTAL kolom y (klasifikasi: X/K per kolom). --lam 1 = tanpa bobot.
    --sem / --no_sem           Stochastic EM (E-step 1 sampel acak). Default MATI.
    --loss_y {mse, ce}         KLASIFIKASI saja: loss blok y. ce = head softmax + cross-entropy (bobot --ce_weight,
                               default = --lam). Fitur selalu MSE EDM. Butuh jaringan dengan penanda c
                               (masking atau PLR nyala).
    --no_masking --no_plr --lam 1          = DiffPuter (N = 10) + pembacaan posterior mean.
Cara baca (keduanya SELALU dihitung untuk ablasi; saklar hanya memilih HASIL UTAMA):
    --readout {pm, sampel}     pm = posterior mean (default); sampel = rata-rata N sampel (cara DiffPuter)
    --hasil {terakhir, ensemble}  terakhir = iterasi EM terakhir (default); ensemble = gabungan semua iterasi
                               (butuh evaluasi tiap iterasi, jadi tidak boleh dengan --eval_last_only)

Metrik (semua di VAL dan TEST, tanpa tambahan lain):
    regresi     : RMSE, MAE, R2 (prediksi titik); dari sampel (10 bila 'terakhir', gabungan bila 'ensemble'):
                  CRPS (rumus fair), coverage interval min-max (ideal (N-1)/(N+1)) dan lebar interval rata-rata.
                  Skala standar-log: z-score log1p(y) dengan statistik train (tanpa log bila log_target false).
    klasifikasi : Accuracy, F1 macro, ROC-AUC, Brier (jumlah atas K kelas; biner juga brier_biner = versi sklearn).
                  Kelas = argmax probabilitas (biner: ambang 0.5). Tanpa pemilihan
                  ambang, tanpa temperature scaling. AUC biner = roc_auc_score(y, p kelas 1); multikelas = OvR macro.
    imputasi    : MAE & RMSE fitur numerik, ACC fitur kategorik (sel fitur yang hilang; cara DiffPuter).
    Langkah sigma posterior mean dipilih di VAL: regresi RMSE terkecil, klasifikasi AUC terbesar.

Keluaran: results_zlog/usulan/{dataset}/{MAR30}/seed{k}/{varian}_*
    _result.txt (per iterasi + HASIL), _metrics.json (semua angka), _{val,test}.csv (prediksi + isian fitur).
    Regresi juga: _{part}_samples_iter{i}.npy, _{part}_pmkurva_iter{i}.npy, _{part}_pm_terakhir.npy,
    _{part}_pm_ensemble.npy (format sama dengan run lama, jadi eval_zlog.py tetap bisa membaca).
    Rekap banyak seed/varian: python rekap_usulan.py
Nama varian otomatis dari komponen yang nyala, contoh:
    default (final)               -> u_plr_mask_lam10_p0.5        (+ _qt bila --num_transform quantile)
    --loss_y ce                   -> u_plr_mask_lam10_p0.5_ce
    --no_plr                      -> u_mask_lam10_p0.5
    --lam 1 --no_plr              -> u_mask_p0.5
    --no_masking --no_plr --lam 1 -> u_dasar
    --sem                         -> u_plr_mask_sem_lam10_p0.5
    --plr_sigma 2                 -> ..._sig2

Kesetaraan dengan skrip lama (teruji): M-step, E-step, sampel & posterior mean IDENTIK dengan run_usulan_final.py
(masking nyala, MSE atau CE) dan run_ablasi.py (masking mati) untuk kombinasi yang sama; yang berubah hanya
metrik yang dilaporkan.

Contoh:
    python metode_usulan.py --dataset bike --ratio 0.3 --seed 0 --parts val test --hapus_ckpt
    python metode_usulan.py --dataset shoppers --ratio 0.3 --seed 2 --parts val test --num_transform quantile --hapus_ckpt
    python metode_usulan.py --dataset shoppers --ratio 0.3 --seed 2 --parts val test --num_transform quantile --loss_y ce
    python metode_usulan.py --dataset bike --ratio 0.3 --seed 0 --parts val test --no_plr          # ablasi tanpa PLR
    python metode_usulan.py --dataset bike --ratio 0.3 --seed 0 --quick                             # uji alur cepat
"""

import os
import math
import json
import time
import shutil

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from data_loader import (load_split, clip_bounds, inverse_transform, model_to_eval, load_ids, inverse_y, y_true_z,
                         y_true_index)
from run_diffputer import (MLPDiffusion, Precond, imputation_metrics, clip_generated, crps_fair,
                           SIGMA_MIN, SIGMA_MAX, RHO, S_CHURN, S_MIN, S_MAX, S_NOISE)
from run_usulan import build_parser, estep_train
from run_usulan_mask import PrecondMask, edm_loss_mask, add_mask_args
from run_usulan_mask import build_net as build_net_mask
from run_usulan_plr_mask_sem_lamfix import MLPDiffusionPLR

GROUP = 'usulan_katemb'
EPS = 1e-6


# ======================================================================
# Jaringan
# ======================================================================
class PrecondMaskCE(PrecondMask):
    """Head kategorik untuk blok y (klasifikasi, --loss_y ce): keluaran mentah F_x kolom y = logit,
    denoiser blok y = softmax(logit) - 0.5 = taksiran E[y_onehot | x_t]. Kolom fitur = PrecondMask."""
    def __init__(self, base, y_cols):
        super().__init__(base.denoise_fn_F, base.sigma_min, base.sigma_max, base.sigma_data)
        self.y_cols = list(y_cols)

    def forward_logits(self, x, sigma, c=None):
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
        x_in = c_in * x * c + x * (1 - c)
        F_x = self.denoise_fn_F(x_in, c, c_noise.flatten()).to(torch.float32)
        D = c_skip * x + c_out * F_x
        logits = F_x[:, self.y_cols]
        D = torch.cat([D[:, :self.y_cols[0]], torch.softmax(logits, dim=1) - 0.5, D[:, self.y_cols[-1] + 1:]], dim=1)
        return D, logits

    def forward(self, x, sigma, c=None):
        return self.forward_logits(x, sigma, c)[0]


class MLPDiffusionPLRKat(MLPDiffusionPLR):
    """MLPDiffusionPLR + embedding kategorik (versi sederhana). cat_info: daftar per fitur kategorik
    {'cols': kolom bit, 'mean','std': standardisasi bit (data_loader), 'n_cat': jumlah kategori}.
    Sel bersih (c = 0): bit -> 0/1 (balik standardisasi, ambang 0.5) -> indeks kategori (kode biner data_loader:
    kategori ke-i = biner i, bit pertama paling kiri) -> embedding. Sel ber-noise (c = 1): bit mentah."""
    def __init__(self, d_in, num_idx, dim_t, n_freq, d_emb, sigma, cat_info):
        super().__init__(d_in, num_idx, dim_t, n_freq, d_emb, sigma)
        self.cat_info = cat_info
        cat_cols = [j for ci in cat_info for j in ci['cols']]
        self.other_idx = [j for j in range(d_in) if j not in set(self.num_idx) and j not in set(cat_cols)]
        self.emb = torch.nn.ModuleList()
        for ci in cat_info:
            e = torch.nn.Embedding(ci['n_cat'], d_emb)
            torch.nn.init.uniform_(e.weight, -1 / math.sqrt(d_emb), 1 / math.sqrt(d_emb))
            self.emb.append(e)
            self.register_buffer(f'kat_mean_{len(self.emb) - 1}', torch.tensor(ci['mean'], dtype=torch.float32))
            self.register_buffer(f'kat_std_{len(self.emb) - 1}', torch.tensor(ci['std'], dtype=torch.float32))
            n_b = len(ci['cols'])
            self.register_buffer(f'kat_pow_{len(self.emb) - 1}',
                                 torch.tensor([2 ** (n_b - 1 - i) for i in range(n_b)], dtype=torch.long))
        n_in = len(self.num_idx) * d_emb + sum(len(ci['cols']) + d_emb for ci in cat_info) + len(self.other_idx) + d_in
        self.proj = torch.nn.Linear(n_in, dim_t)

    def forward(self, x, c, noise_labels):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        parts = [self.plr(x[:, self.num_idx])]
        for j, ci in enumerate(self.cat_info):
            cols = ci['cols']
            xb = x[:, cols]
            cj = c[:, cols[0]:cols[0] + 1]                                       # mask sama untuk semua bit
            bit = (xb * 2 * getattr(self, f'kat_std_{j}') + getattr(self, f'kat_mean_{j}') > 0.5).long()
            k = (bit * getattr(self, f'kat_pow_{j}')).sum(1).clamp(max=ci['n_cat'] - 1)
            parts.append(self.emb[j](k) * (1 - cj))                              # bersih -> embedding
            parts.append(xb * cj)                                                # ber-noise -> bit mentah
        h = torch.cat(parts + [x[:, self.other_idx], c], dim=1)
        return self.mlp(self.proj(h) + emb)


def build_net(dim, args, device, print_model):
    if args.plr:                                                    # PLR (+ penanda c)
        num_idx = list(range(args.n_num)) + ([args.y_cols[0]] if args.task == 'regression' else [])
        if args.emb_kategorik:
            denoise_fn = MLPDiffusionPLRKat(dim, num_idx, args.hid_dim, args.plr_k, args.plr_dim, args.plr_sigma,
                                            args.cat_info).to(device)
        else:
            denoise_fn = MLPDiffusionPLR(dim, num_idx, args.hid_dim, args.plr_k, args.plr_dim,
                                         args.plr_sigma).to(device)
        if print_model:
            print(denoise_fn)
            if args.emb_kategorik:
                print(f'[EMBEDDING KATEGORIK] {len(args.cat_info)} fitur, kategori '
                      f'{[ci["n_cat"] for ci in args.cat_info]}, d = {args.plr_dim} (sel bersih; sel ber-noise = bit)')
            print(f'[PLR] {len(num_idx)} kolom numerik lewat PLR; {len(denoise_fn.other_idx)} kolom apa adanya; '
                  f'parameter = {sum(p.numel() for p in denoise_fn.parameters()):,}')
        net = PrecondMask(denoise_fn).to(device)
    elif args.masking:                                              # MLP input [x, c]
        net = build_net_mask(dim, args, device, print_model)
    else:                                                           # DiffPuter asli, tanpa c
        denoise_fn = MLPDiffusion(dim, args.hid_dim).to(device)
        if print_model:
            print(denoise_fn)
        net = Precond(denoise_fn).to(device)
    if args.loss_y == 'ce':
        net = PrecondMaskCE(net, args.y_cols).to(device)
    return net


# ======================================================================
# Loss & M-step
# ======================================================================
def make_cond_y(mask_rows, p, y_cols):
    """Penanda c: dengan peluang p sebuah baris kondisional (fitur teramati bersih, c = 0; fitur bolong & semua
    kolom y ditebak, c = 1); sisanya full noise (c = 1 semua)."""
    B, d = mask_rows.shape
    c_cond = mask_rows.clone()
    c_cond[:, y_cols] = 1.0
    is_cond = (torch.rand(B, 1, device=mask_rows.device) < p).float()
    return is_cond * c_cond + (1 - is_cond) * torch.ones_like(c_cond)


def edm_loss_full(net, data, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """Loss EDM DiffPuter (semua kolom diberi noise), error PER SEL."""
    sigma = (torch.randn(data.shape[0], device=data.device) * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(data) * sigma.unsqueeze(1)
    D = net(data + n, sigma)
    return weight.unsqueeze(1) * (D - data) ** 2


def edm_loss_ce(net, data, c, y_cols, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    """Fitur: loss EDM (urutan acak = edm_loss_mask). Blok y: cross-entropy logit vs kelas benar.
    Kembali (error fitur per sel dengan kolom y = 0, CE per baris)."""
    sigma = (torch.randn(data.shape[0], device=data.device) * P_std + P_mean).exp()
    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    n = torch.randn_like(data) * sigma.unsqueeze(1) * c
    D, logits = net.forward_logits(data + n, sigma, c)
    err = weight.unsqueeze(1) * (D - data) ** 2 * c
    err[:, y_cols] = 0.0
    ce = torch.nn.functional.cross_entropy(logits, data[:, y_cols].argmax(1), reduction='none')
    return err, ce


def train_mstep(train_data, train_mask, dim, args, device, ckpt_iter_dir, print_model):
    """M-step. Loss = (sum lambda(sigma) * w_kolom * c * (D - x)^2 [+ ce_weight * sum CE_y]) / sum c,
    early stopping dari loss tanpa bobot kolom."""
    X = torch.tensor(train_data, dtype=torch.float32, device=device)
    M = torch.tensor(train_mask, dtype=torch.float32, device=device)
    net = build_net(dim, args, device, print_model)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)
    col_w = torch.ones(dim, device=device)
    col_w[args.y_cols] = args.lam / len(args.y_cols)

    best, patience = float('inf'), 0
    n = X.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        perm = torch.randperm(n, device=device)
        tot_err, tot_cnt = 0.0, 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            if args.masking:
                c = make_cond_y(M[idx], args.p_cond, args.y_cols)
            if args.loss_y == 'ce':
                if not args.masking:
                    c = torch.ones_like(X[idx])
                err, ce = edm_loss_ce(net, X[idx], c, args.y_cols)
                cnt = c.sum()
                loss = (err.sum() + args.ce_weight * ce.sum()) / cnt
                err_log = err.detach().sum().item() + ce.detach().sum().item()
            else:
                if args.masking:
                    err = edm_loss_mask(net, X[idx], c)
                    cnt = c.sum()
                else:
                    err = edm_loss_full(net, X[idx])
                    cnt = torch.tensor(float(err.numel()), device=device)
                loss = (err * col_w).sum() / cnt
                err_log = err.detach().sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_err += err_log
            tot_cnt += cnt.item()
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
    return net


# ======================================================================
# Sampling uji + rekam output denoiser kolom y (untuk posterior mean)
# ======================================================================
def _t_steps(net, num_steps, device):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min, sigma_max = max(SIGMA_MIN, net.sigma_min), min(SIGMA_MAX, net.sigma_max)
    t = (sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) *
         (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    return torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])])


@torch.no_grad()
def impute_cond_record(net, x, mask, y_cols, num_steps, resample):
    """Sampling KONDISIONAL (masking nyala): nilai teramati bersih, sel c = 1 ditebak. Rekam denoiser kolom y."""
    t_steps = _t_steps(net, num_steps, x.device)
    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1)
    c = mask.to(torch.float32)
    x_t = x * (1 - c) + torch.randn_like(x) * t_steps[0] * c
    rec = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        reps = resample if i < num_steps - 1 else 1
        for j in range(reps):
            x_cur = x_t
            g = gamma if S_MIN <= t_cur <= S_MAX else 0
            t_hat = net.round_sigma(t_cur + g * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur) * c
            denoised = net(x_hat, t_hat, c).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < num_steps - 1:
                den2 = net(x_next, t_next, c).to(torch.float32)
                d_prime = (x_next - den2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_new = x * (1 - c) + x_next * c
            if j == reps - 1:
                x_t = x_new
                rec.append(denoised[:, y_cols].cpu().numpy())
            else:
                x_t = x_new + torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt() * c
    return x_t, np.stack(rec, 0), (t_steps[:-1] * (1 + gamma)).cpu().numpy()


@torch.no_grad()
def impute_full_record(net, x, mask, y_cols, num_steps, resample):
    """Sampling FULL NOISE + penimpaan nilai teramati (cara DiffPuter, masking mati). Rekam denoiser kolom y."""
    t_steps = _t_steps(net, num_steps, x.device)
    gamma = min(S_CHURN / num_steps, np.sqrt(2) - 1)
    mask = mask.to(torch.float32)
    x_t = torch.randn_like(x) * t_steps[0]
    rec = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        reps = resample if i < num_steps - 1 else 1
        for j in range(reps):
            x_known = x + torch.randn_like(x) * t_next
            x_cur = x_t
            g = gamma if S_MIN <= t_cur <= S_MAX else 0
            t_hat = net.round_sigma(t_cur + g * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_NOISE * torch.randn_like(x_cur)
            denoised = net(x_hat, t_hat).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < num_steps - 1:
                den2 = net(x_next, t_next).to(torch.float32)
                d_prime = (x_next - den2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_t_prev = x_known * (1 - mask) + x_next * mask
            if j == reps - 1:
                x_t = x_t_prev
                rec.append(denoised[:, y_cols].cpu().numpy())
            else:
                x_t = x_t_prev + torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()
    return x_t, np.stack(rec, 0), (t_steps[:-1] * (1 + gamma)).cpu().numpy()


@torch.no_grad()
def run_trials_record(net, X_obs, mask, args, device, desc, bounds):
    """N lintasan. Kembali S [N, n, d] (di-clip), D [N, steps, n, K kolom y], sigma per langkah."""
    impute = impute_cond_record if args.masking else impute_full_record
    recs, Ds, sig = [], [], None
    for _ in tqdm(range(args.num_trials), desc=desc):
        out = np.empty_like(X_obs, dtype=np.float32)
        outD = []
        for s in range(0, X_obs.shape[0], args.chunk_rows):
            xb = torch.tensor(X_obs[s:s + args.chunk_rows], dtype=torch.float32, device=device)
            mb = torch.tensor(mask[s:s + args.chunk_rows], dtype=torch.float32, device=device)
            rec, D, sig = impute(net, xb, mb, args.y_cols, args.num_steps, args.resample_test)
            out[s:s + args.chunk_rows] = (rec * mb + xb * (1 - mb)).cpu().numpy()
            outD.append(D)
        recs.append(out)
        Ds.append(np.concatenate(outD, axis=1))
    return clip_generated(np.stack(recs, axis=0), mask, bounds, desc), np.stack(Ds, 0), sig


# ======================================================================
# Pembacaan & metrik
# ======================================================================
def probs_from_samples(Y):
    """Sampel blok one-hot [N, n, K] (+-0.5) -> probabilitas [n, K] (rata-rata sampel, cara DiffPuter)."""
    q = np.clip(Y.mean(0) + 0.5, EPS, 1.0)
    return q / q.sum(1, keepdims=True)


def probs_posterior_mean(D):
    """Rekaman denoiser blok y [N, steps, n, K] -> probabilitas per langkah [steps, n, K] (posterior mean)."""
    q = np.clip(D.mean(0) + 0.5, EPS, 1.0)
    return q / q.sum(2, keepdims=True)


def reg_metrics(y, pred, samples):
    """Regresi: RMSE, MAE, R2 dari prediksi titik; dari sampel [N, n]: CRPS (rumus fair/tak bias),
    coverage interval min-max (ideal (N-1)/(N+1): 0.818 bila N = 10, 0.961 bila N = 50) dan lebar rata-ratanya."""
    e = y - pred
    mn, mx = samples.min(0), samples.max(0)
    N = samples.shape[0]
    return {'rmse': float(np.sqrt((e ** 2).mean())), 'mae': float(np.abs(e).mean()),
            'r2': float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), 'crps': crps_fair(samples, y),
            'cov': float(((y >= mn) & (y <= mx)).mean()), 'cov_ideal': (N - 1) / (N + 1),
            'lebar': float((mx - mn).mean()), 'n_sampel': int(N)}


def cls_metrics(P, y):
    """Klasifikasi: Accuracy, F1 macro, ROC-AUC, Brier. Kelas = argmax (biner = ambang 0.5).
    Brier = versi asli Brier (1950): jumlah (p_k - onehot_k)^2 atas K kelas, rata-rata baris (sama dengan
    metrics_cls.py / rekap lama). Biner: 'brier_biner' = rata-rata (p_1 - y)^2 (= brier_score_loss sklearn = brier/2)."""
    K = P.shape[1]
    pred = P.argmax(1)
    try:
        auc = roc_auc_score(y, P[:, 1]) if K == 2 else roc_auc_score(y, P, multi_class='ovr', average='macro',
                                                                     labels=list(range(K)))
    except ValueError:                                   # ada kelas yang tidak muncul di bagian ini
        auc = float('nan')
    out = {'acc': float((pred == y).mean()),
           'f1_macro': float(f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0)),
           'auc': float(auc), 'brier': float(((P - np.eye(K)[y]) ** 2).sum(1).mean())}
    if K == 2:
        out['brier_biner'] = float(((P[:, 1] - y) ** 2).mean())
    return out


def pilih_langkah(curve_val, y_val, is_reg):
    """Langkah sigma posterior mean di VAL: regresi RMSE terkecil, klasifikasi AUC terbesar."""
    if is_reg:
        return int(np.argmin([np.sqrt(((y_val - c) ** 2).mean()) for c in curve_val]))
    s = np.nan_to_num(np.array([cls_metrics(c, y_val)['auc'] for c in curve_val]), nan=-np.inf)
    return int(np.argmax(s))


def fmt(m, is_reg):
    if is_reg:
        return (f'RMSE {m["rmse"]:.4f} MAE {m["mae"]:.4f} R2 {m["r2"]:.4f} CRPS {m["crps"]:.4f} '
                f'Cov {m["cov"]:.3f} (ideal {m["cov_ideal"]:.3f}) Lebar {m["lebar"]:.4f}')
    return f'Acc {m["acc"]:.4f} F1(macro) {m["f1_macro"]:.4f} AUC {m["auc"]:.4f} Brier {m["brier"]:.4f}'


def simpan_csv(path, part, d, S, meta, args, kolom):
    """CSV: id, fitur (NaN bila di-mask), {fitur}_isian (rata-rata sampel), y_asli, kolom prediksi."""
    df = d['df']
    feat_cols = meta['num_cols'] + meta['cat_cols']
    out = pd.DataFrame({'id': load_ids(f'datasets/{args.dataset}/splits/seed{args.seed}/{part}_idx.npy', len(df))})
    shown = df[feat_cols].mask(d['mask_feat'])
    imputed = inverse_transform(model_to_eval(S.mean(0), meta), meta)
    for j, c in enumerate(feat_cols):
        out[c] = shown[c].values
        if d['mask_feat'][:, j].any():
            out[f'{c}_isian'] = np.where(d['mask_feat'][:, j], imputed[c].values, None)
    out['y_asli'] = y_true_z(df, meta) if args.task == 'regression' else df[meta['target']].values
    for k, v in kolom.items():
        out[k] = v
    out.to_csv(path, index=False)


# ======================================================================
# Alur EM (regresi & klasifikasi)
# ======================================================================
def run(args):
    if args.quick:
        args.max_iter, args.max_epochs, args.patience = 2, 200, 50
        args.num_steps, args.resample, args.num_trials = 10, 2, 5
        args.varian += '_quick'
    if args.resample_test is None:
        args.resample_test = args.resample
    parts = ['val'] + [p for p in args.parts if p != 'val']           # val dibutuhkan memilih langkah sigma
    args.parts = parts
    is_reg = args.task == 'regression'

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tag = f'{args.mech}{int(round(args.ratio * 100)):02d}'
    ckpt_dir = f'ckpt/{GROUP}/{args.dataset}/{tag}/seed{args.seed}/{args.varian}/{args.num_trials}_{args.num_steps}'
    pred_dir = f'{args.out_dir}/{GROUP}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    prefix = f'{pred_dir}/{args.varian}'
    result_path = f'{prefix}_result.txt'
    open(result_path, 'w').close()

    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech, num_transform=args.num_transform)
    dim, yi = meta['dim'], meta['y_idx']
    X_tr, M_tr = data['train']['X_obs'], data['train']['mask']
    y_true = {p: (y_true_z(data[p]['df'], meta) if is_reg else y_true_index(data[p]['df'], meta)) for p in parts}
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | {args.task} | device {device} | '
          f'train {X_tr.shape}, val {data["val"]["X_obs"].shape}, test {data["test"]["X_obs"].shape}')
    print(f'Varian {args.varian} | {args.info}')

    summary = {'varian': args.varian, 'task': args.task, 'dataset': args.dataset, 'mech': args.mech,
               'ratio': args.ratio, 'seed': args.seed, 'komponen': args.info, 'args': vars(args), 'iterasi': []}
    REC = {'sampel': {p: [] for p in parts}, 'pm': {p: [] for p in parts}}   # per iterasi yang dievaluasi
    S_last, iters = {}, []
    t_all = time.time()

    for it in range(args.max_iter):
        print('=' * 60)
        print(f'Iteration: {it}')
        os.makedirs(f'{ckpt_dir}/{it}', exist_ok=True)
        X_miss = X_tr.copy() if it == 0 else np.load(f'{ckpt_dir}/iter_{it}.npy')

        t0 = time.time()                                              # ---- M-step ----
        net = train_mstep(X_miss, M_tr, dim, args, device, f'{ckpt_dir}/{it}', print_model=(it == 0))
        res = {'iterasi': it, 'waktu_train_s': time.time() - t0}

        if M_tr.any():                                                # ---- E-step (data train) ----
            t0 = time.time()
            rec_X = estep_train(net, X_tr, M_tr, args, device, clip_bounds(meta))
            np.save(f'{ckpt_dir}/iter_{it + 1}.npy', rec_X)
            mae, rmse, acc, _ = imputation_metrics(rec_X, data['train']['X_true'], data['train']['mask_feat'], meta)
            res['imputasi_train'] = {'mae': mae, 'rmse': rmse, 'acc': acc, 'waktu_s': time.time() - t0}
            print(f'In-sample → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}')
        else:
            np.save(f'{ckpt_dir}/iter_{it + 1}.npy', X_tr)

        if it == args.max_iter - 1 or not args.eval_last_only:       # ---- evaluasi val & test ----
            iters.append(it)
            for part in parts:
                d = data[part]
                t0 = time.time()
                S, D, sig = run_trials_record(net, d['X_obs'], d['mask'], args, device, f'Sampling {part}',
                                              clip_bounds(meta))
                mae, rmse, acc, _ = imputation_metrics(S.mean(0), d['X_true'], d['mask_feat'], meta)
                res[part] = {'imputasi': {'mae': mae, 'rmse': rmse, 'acc': acc}, 'waktu_s': time.time() - t0}
                if is_reg:
                    samp = inverse_y(S[:, :, yi], meta)                 # [N, n] skala standar-log
                    curve = inverse_y(D[..., 0].mean(0), meta)          # [steps, n] posterior mean
                    np.save(f'{prefix}_{part}_samples_iter{it}.npy', samp.T.astype(np.float32))
                    np.save(f'{prefix}_{part}_pmkurva_iter{it}.npy', curve.astype(np.float32))
                else:
                    samp = probs_from_samples(S[:, :, args.y_cols])     # [n, K]
                    curve = probs_posterior_mean(D)                     # [steps, n, K]
                REC['sampel'][part].append(samp)
                REC['pm'][part].append(curve)
                S_last[part] = S
            s_it = pilih_langkah(REC['pm']['val'][-1], y_true['val'], is_reg)
            with open(result_path, 'a') as f:
                for part in parts:
                    samp, curve, y = REC['sampel'][part][-1], REC['pm'][part][-1], y_true[part]
                    if is_reg:
                        m_s = reg_metrics(y, samp.mean(0), samp)
                        m_p = reg_metrics(y, curve[s_it], samp)
                    else:
                        m_s, m_p = cls_metrics(samp, y), cls_metrics(curve[s_it], y)
                    res[part].update({'sampel': m_s, 'pm': {**m_p, 'langkah': s_it, 'sigma': float(sig[s_it])}})
                    im = res[part]['imputasi']
                    line = (f'iteration {it} | {part} | imputasi MAE {im["mae"]:.6f} RMSE {im["rmse"]:.6f} '
                            f'ACC {im["acc"]} | sampel {fmt(m_s, is_reg)} | pm (sigma {sig[s_it]:.3g}) '
                            f'{fmt(m_p, is_reg)}')
                    print(line)
                    f.write(line + '\n')
        summary['iterasi'].append(res)
        with open(f'{prefix}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

    # ---- HASIL AKHIR: semua cara baca (untuk ablasi); HASIL UTAMA = --readout x --hasil ----
    hasil_list = ['terakhir'] + (['ensemble'] if len(iters) > 1 else [])
    summary['hasil'], summary['utama'] = {}, {'readout': args.readout, 'hasil': args.hasil}
    titik = {p: {} for p in parts}
    print('=' * 60)
    with open(result_path, 'a') as f:
        for h in hasil_list:
            if is_reg:
                samp_h = {p: REC['sampel'][p][-1] if h == 'terakhir' else np.concatenate(REC['sampel'][p], 0)
                          for p in parts}
            else:
                samp_h = {p: REC['sampel'][p][-1] if h == 'terakhir' else np.mean(REC['sampel'][p], 0) for p in parts}
            curve_h = {p: REC['pm'][p][-1] if h == 'terakhir' else np.mean(REC['pm'][p], 0) for p in parts}
            s_h = pilih_langkah(curve_h['val'], y_true['val'], is_reg)
            for r in ['pm', 'sampel']:
                kunci = f'{r}_{h}'
                summary['hasil'][kunci] = {'langkah_pm': s_h, 'sigma_pm': float(sig[s_h]),
                                           'iterasi': iters if h == 'ensemble' else [iters[-1]]}
                for p in parts:
                    y = y_true[p]
                    if is_reg:
                        pt = curve_h[p][s_h] if r == 'pm' else samp_h[p].mean(0)
                        m = reg_metrics(y, pt, samp_h[p])
                    else:
                        pt = curve_h[p][s_h] if r == 'pm' else samp_h[p]
                        m = cls_metrics(pt, y)
                    titik[p][kunci] = pt
                    summary['hasil'][kunci][p] = m
                    tanda = '  <- UTAMA' if (r, h) == (args.readout, args.hasil) else ''
                    line = f'HASIL {r} x {h} | {p} | {fmt(m, is_reg)}{tanda}'
                    print(line)
                    f.write(line + '\n')
    utama = f'{args.readout}_{args.hasil}'
    for p in parts:
        summary['utama'][p] = summary['hasil'][utama][p]
        if is_reg:
            kolom = {'y_pred': titik[p][utama], **{f'y_pred_{k}': v for k, v in titik[p].items()}}
            for k in ('pm_terakhir', 'pm_ensemble'):
                if k in titik[p]:
                    np.save(f'{prefix}_{p}_{k}.npy', titik[p][k].astype(np.float32))
        else:
            P = titik[p][utama]
            kolom = {'y_pred': np.array(meta['classes'], dtype=object)[P.argmax(1)],
                     **{f'p_{cl}': P[:, k] for k, cl in enumerate(meta['classes'])}}
        simpan_csv(f'{prefix}_{p}.csv', p, data[p], S_last[p], meta, args, kolom)
    summary['waktu_total_s'] = time.time() - t_all
    with open(f'{prefix}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Selesai ({summary["waktu_total_s"]:.0f}s). Hasil: {prefix}_*  | checkpoint: {ckpt_dir}')
    return ckpt_dir


# ======================================================================
def nama_varian(args):
    on = (['plr'] if args.plr else []) + (['mask'] if args.masking else []) + (['sem'] if args.sem else []) \
        + ([f'lam{args.lam:g}'] if args.lam != 1 else [])
    nama = 'u_' + ('_'.join(on) if on else 'dasar')
    if args.masking:
        nama += f'_p{args.p_cond:g}'
    if args.num_transform != 'none':
        nama += {'quantile': '_qt', 'yeojohnson': '_yj'}[args.num_transform]
    if args.loss_y == 'ce':
        nama += '_ce' + (f'{args.ce_weight:g}' if args.ce_weight != args.lam else '')
    if args.plr:
        nama += (f'_sig{args.plr_sigma:g}' if args.plr_sigma != 1.0 else '') + \
                (f'_k{args.plr_k}' if args.plr_k != 48 else '') + (f'_d{args.plr_dim}' if args.plr_dim != 24 else '')
    if args.emb_kategorik:
        nama += '_katemb'
    return nama


def main():
    p = add_mask_args(build_parser('Metode usulan (regresi & klasifikasi), komponen bongkar pasang'))
    for k, teks in [('masking', 'target-aware masking + penanda c'), ('plr', 'embedding PLR'),
                    ('sem', 'Stochastic EM')]:
        p.add_argument(f'--{k}', dest=k, action='store_true', help=f'nyalakan {teks}')
        p.add_argument(f'--no_{k}', dest=k, action='store_false', help=f'matikan {teks}')
    p.set_defaults(masking=True, plr=True, sem=False, num_trials=10)
    p.add_argument('--lam', type=float, default=10.0, help='bobot TOTAL kolom y (klasifikasi: lam/K per kolom); 1 = tanpa')
    p.add_argument('--plr_k', type=int, default=48)
    p.add_argument('--plr_dim', type=int, default=24)
    p.add_argument('--plr_sigma', type=float, default=1.0)
    p.add_argument('--emb_kategorik', action='store_true',
                   help='embedding kategorik untuk sel bersih (sejajar PLR); sel ber-noise tetap bit mentah')
    p.add_argument('--loss_y', default='mse', choices=['mse', 'ce'], help='KLASIFIKASI: loss blok y')
    p.add_argument('--ce_weight', type=float, default=None, help='bobot CE per baris (default = --lam)')
    p.add_argument('--readout', default='pm', choices=['pm', 'sampel'], help='prediksi titik hasil utama')
    p.add_argument('--hasil', default='terakhir', choices=['terakhir', 'ensemble'], help='iterasi untuk hasil utama')
    p.add_argument('--task', default='auto', choices=['auto', 'regression', 'classification'])
    p.add_argument('--num_transform', default='none', choices=['none', 'quantile', 'yeojohnson'],
                   help="transformasi fitur numerik (fit di nilai teramati train); 'none' = tanpa")
    p.add_argument('--hapus_ckpt', action='store_true', help='hapus checkpoint varian ini setelah selesai')
    args = p.parse_args()

    _, meta = load_split(args.dataset, args.seed, args.ratio, args.mech)        # struktur kolom saja
    if args.task == 'auto':
        args.task = meta['task']
    elif args.task != meta['task']:
        raise SystemExit(f'--task {args.task} tidak cocok dengan task_type "{meta["task"]}" di datasets/Info.')
    args.n_num, args.y_cols = meta['n_num'], meta['y_cols']
    args.cat_info, start = [], meta['n_num']                          # struktur bit kategorik (data_loader)
    for c_, nb in zip(meta['cat_cols'], meta['cat_bits']):
        cols = list(range(start, start + nb))
        args.cat_info.append({'nama': c_, 'cols': cols, 'n_cat': len(meta['cat_maps'][c_]['categories']),
                              'mean': [float(meta['mean'][j]) for j in cols],
                              'std': [float(meta['std'][j]) for j in cols]})
        start += nb
    if args.ce_weight is None:
        args.ce_weight = args.lam
    if args.loss_y == 'ce' and args.task != 'classification':
        raise SystemExit('--loss_y ce hanya untuk klasifikasi (regresi selalu MSE).')
    if args.loss_y == 'ce' and not (args.masking or args.plr):
        raise SystemExit('--loss_y ce butuh jaringan dengan penanda c (masking atau PLR nyala).')
    if args.emb_kategorik and not (args.plr and args.masking):
        raise SystemExit('--emb_kategorik butuh PLR dan masking nyala.')
    if args.emb_kategorik and not args.cat_info:
        raise SystemExit('--emb_kategorik: dataset ini tidak punya fitur kategorik.')
    if args.hasil == 'ensemble' and args.eval_last_only:
        raise SystemExit('--hasil ensemble butuh evaluasi tiap iterasi: jangan pakai --eval_last_only.')
    if args.varian == 'auto':
        args.varian = nama_varian(args)
    nyala = [n for n, on in [('masking', args.masking), ('PLR', args.plr), ('SEM', args.sem),
                             (f'lambda {args.lam:g}', args.lam != 1)] if on]
    args.info = (f'komponen: {" + ".join(nyala) or "dasar DiffPuter"} | E-step: '
                 f'{"SEM 1 sampel" if args.sem else f"rata-rata {args.num_trials} sampel"} | sampling uji: '
                 f'{"kondisional" if args.masking else "full noise (DiffPuter)"} | fitur numerik: {args.num_transform}'
                 + (f' | PLR sigma {args.plr_sigma:g}, K {args.plr_k}, d {args.plr_dim}' if args.plr else '')
                 + (' | EMBEDDING KATEGORIK (sel bersih)' if args.emb_kategorik else '')
                 + (f' | loss y: {args.loss_y}' if args.task == 'classification' else '')
                 + f' | hasil utama: {args.readout} x {args.hasil}')

    ckpt_dir = run(args)
    if args.hapus_ckpt:
        ck = os.path.dirname(ckpt_dir)
        if os.path.isdir(ck):
            shutil.rmtree(ck)
            print(f'[hapus_ckpt] checkpoint dihapus: {ck}')


if __name__ == '__main__':
    main()
