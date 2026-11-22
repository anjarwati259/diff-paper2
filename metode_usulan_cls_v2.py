"""
METODE USULAN - KLASIFIKASI ALUR STANDAR v2 (20 Nov 2026) = metode_usulan_cls_standar.py (md5 7143a711...) + 3 saklar
baru, SEMUA DEFAULT MATI (tanpa saklar = training, RNG, hasil dan nama varian IDENTIK versi sebelumnya).

MASALAH yang disasar: noise level sigma saat latihan ~ lognormal(P_mean -1.2, P_std 1.2) (median 0.30). Blok y one-hot
{0,1}: pada sigma < ~1 kelas masih terbaca dari input y yang ber-noise (galat baca 0.9% di sigma 0.3, 24% di sigma 1),
sehingga CE di sebagian besar baris = "salin label dari input" (mudah, ~0) dan TIDAK melatih p(y | fitur teramati).
Hanya 2.8% undian sigma >= 3 (y praktis tak terbaca) dan x p_cond 0.5 -> ~1.4% baris benar-benar melatih tugas prediksi.
Padahal sampler mulai dari sigma 80 dan 24 dari 50 langkah ada di sigma >= 3 (rezim yang hampir tidak pernah dilatih;
P(sigma >= 10) = 0.18%). Langkah pm terpilih di val pun sering di sigma besar (0.7-40).

  --ce_hi W  (default 0 = mati): CE TAMBAHAN di sigma tinggi. Tiap batch, satu forward tambahan: semua baris dibuat
        KONDISIONAL (fitur teramati bersih c = 0; fitur bolong & blok y c = 1), sigma ~ log-uniform[--ce_hi_min (3),
        --ce_hi_max (80)], loss = CE logit y vs kelas benar (fitur tidak diberi loss di forward ini).
        Loss total = (loss lama + W * sum CE_hi) / sum c  -> W sebanding dengan --ce_weight (10).
        = melatih langsung p(y | x_obs, pola missing) di rezim tempat sampler mulai (keadaan y belum diketahui).
        Loss EDM fitur, CE lama, sampler, cara baca TIDAK berubah. Nama varian + '_hi{W}' (+ '_m{min}' bila min != 3).
  --probe_every E (default 0 = mati): tiap E epoch, probe VAL murah: 1 forward kondisional di sigma --probe_sigmas
        (80 10 3 1) -> CE & AUC val per sigma. Dicatat ke metrics.json ('probe' per iterasi) sebagai DIAGNOSA
        (apakah head y overfit selama M-step?). Noise probe dari torch.Generator terpisah -> training TIDAK berubah.
  --val_ckpt : checkpoint M-step = epoch dengan CE val probe (rata-rata atas --probe_sigmas) terkecil, bukan loss
        train terkecil. Early stopping tetap dari loss train. Butuh probe (bila --probe_every 0 -> dipakai 50).
        Nama varian + '_vck'. Val sudah dipakai memilih langkah pm, jadi tidak ada kebocoran test.

  --pilih_pm {auc, logloss} (tambahan 20 Nov; default auc = versi lama): kriteria memilih langkah sigma pm di VAL.
        AUC hanya menilai urutan -> bisa memilih langkah sigma besar yang peluangnya menciut ke prior; argmax lalu
        selalu kelas mayoritas (F1 macro ~0.46 di Shoppers; terlihat di MAR70). logloss = CE val (proper scoring
        rule, sama dengan loss training) menilai urutan DAN besar peluang. Nama varian + '_ll'.
        Klasifikasi kini juga menyimpan kurva pm {prefix}_{val,test}_pmkurva_iter{it}.npy [langkah, n, K] -> kriteria
        bisa dihitung ulang tanpa run (pilih_ulang_pm.py). Hasil & angka lain TIDAK berubah.
  Penyimpanan peluang PER LINTASAN (tambahan 21 Nov; default NYALA, matikan dengan --tanpa_lintasan): klasifikasi
        menyimpan {prefix}_{val,test}_lintasan_iter{it}.npy = softmax blok y tiap lintasan sampling & tiap langkah
        sigma, bentuk [N lintasan, langkah, n, K], float16 (Shoppers ~5 MB test + ~2.5 MB val per iterasi). Untuk
        analisis ketidakpastian per baris ala CARD (Han dkk., NeurIPS 2022): sebaran peluang antar lintasan, benar vs
        salah, prediksi selektif, kaitan dengan jumlah fitur hilang. Rata-rata atas lintasan = pmkurva. Training,
        sampling, dan semua angka hasil TIDAK berubah.

  --ema D (tambahan 21 Nov; default 0 = mati): EMA bobot jaringan di dalam M-step (praktik standar model difusi:
        EDM, Karras dkk. NeurIPS 2022; panjang EMA penting, Karras dkk. CVPR 2024). Setelah tiap langkah optimizer:
        theta_ema <- d * theta_ema + (1 - d) * theta, d = min(D, (1 + k) / (10 + k)) (k = jumlah langkah; pemanasan
        agar EMA tidak terseret bobot acak awal saat langkah per epoch sedikit). Early stopping tetap dari loss train jaringan biasa, tetapi
        checkpoint yang DISIMPAN & DIPAKAI (E-step, sampling uji, pm) = bobot EMA pada epoch terbaik. Probe val
        (bila nyala) juga memakai bobot EMA. Nama varian + '_ema{D}'. Bukan ensemble iterasi EM.
  --warm_start (tambahan 21 Nov): M-step iterasi k >= 1 MULAI dari bobot hasil iterasi k-1 (yang dipakai di E-step),
        bukan inisialisasi acak baru. = bentuk Generalized EM (Dempster, Laird & Rubin 1977): M-step memperbarui
        parameter dari nilai sebelumnya. Optimizer & penjadwal lr dibuat baru tiap iterasi. Nama varian + '_ws'.
  Keduanya default mati -> training, RNG, hasil & nama varian identik versi sebelumnya.

Contoh v2:
    python metode_usulan_cls_v2.py --dataset shoppers --ratio 0.3 --seed 0 --parts val test --probe_every 50 --hapus_ckpt
    python metode_usulan_cls_v2.py --dataset shoppers --ratio 0.3 --seed 0 --parts val test --ce_hi 10 --probe_every 50 --hapus_ckpt
    python metode_usulan_cls_v2.py --dataset shoppers --ratio 0.3 --seed 0 --parts val test --ce_hi 10 --val_ckpt --hapus_ckpt
------------------------------------------------------------------------------------------------------------------
Docstring versi sebelumnya:
METODE USULAN - KLASIFIKASI ALUR STANDAR (logit -> softmax -> cross-entropy; prediksi softmax -> argmax).
Dibuat dari metode_usulan_katemb.py (16 Nov 2026). Hasil di folder TERPISAH: results_zlog/usulan_cls_standar/...
Rekap: python rekap_usulan.py --root results_zlog/usulan_cls_standar

KLASIFIKASI (yang berubah dibanding metode final):
  Data     : blok y = one-hot {0, 1} (K kolom), BUKAN {-0.5, +0.5}.
  Latihan  : keluaran mentah jaringan untuk kolom y = LOGIT -> softmax -> cross-entropy terhadap kelas benar.
             Bobot CE = --ce_weight per baris (default = --lam = 10). Fitur tetap loss MSE EDM (masking + penanda c).
             Tidak ada argmax pada tebakan di loss (argmax(1) pada one-hot LABEL hanya mengubah label ke indeks kelas).
  Prediksi : --num_trials (10) ulangan sampling x --num_steps (50) langkah. Di setiap langkah peluang = softmax(logit),
             SEKALI saja. Peluang = rata-rata softmax dari 10 ulangan; kelas = argmax.
             Tanpa +0.5, tanpa clip/normalisasi, tanpa softmax kedua / suhu T.
  Dua cara baca (keduanya selalu dihitung dan dicetak):
     akhir : peluang di langkah TERAKHIR sampler (alur standar; setara rata-rata sampel akhir cara DiffPuter)
             -> HASIL UTAMA default (--readout akhir)
     pm    : peluang di langkah sigma yang DIPILIH DI VAL (AUC terbesar) = posterior mean (usulan); --readout pm
     Catatan: di VAL, pm >= akhir karena langkahnya dipilih di val. Bandingkan keduanya di TEST, beberapa seed.

REGRESI: tidak berubah (identik metode final: loss MSE, cara baca pm & sampel).
Komponen lain = metode final: target-aware masking (p 0.5) + penanda c, lambda 10, PLR (K 48, d 24, sigma 1),
5 iterasi EM, E-step rata-rata 10 sampel (cara DiffPuter). Saklar ablasi --no_masking / --no_plr / --lam / --sem tetap ada.
Klasifikasi butuh penanda c (masking atau PLR nyala), sama seperti --loss_y ce di skrip lama.

Metrik klasifikasi (VAL & TEST): Accuracy, F1 macro, ROC-AUC, Brier (+ brier_biner); imputasi fitur MAE/RMSE/ACC.
Hasil per iterasi: baris 'akhir' dan 'pm'. Akhir run: 'HASIL {akhir|pm} x {terakhir|ensemble}' (ensemble = rata-rata
peluang semua iterasi EM; hanya bila tanpa --eval_last_only).
Keluaran: results_zlog/usulan_cls_standar/{dataset}/{MAR30}/seed{k}/{varian}_*
    _result.txt, _metrics.json, _{val,test}.csv (y_pred + p_{kelas} dari cara baca UTAMA).
Nama varian: klasifikasi u_plr_mask_lam10_p0.5_ce01 ; regresi u_plr_mask_lam10_p0.5 (sama dengan final).

--plr_y (KLASIFIKASI, default MATI = identik versi sebelumnya; tambahan 18 Nov 2026):
    K kolom blok y ikut lewat PLR (sisi INPUT saja), sama seperti y regresi yang selalu lewat PLR. Di dalam jaringan
    blok y selalu ber-noise (c = 1) sehingga nilainya kontinu. Keluaran jaringan (K logit), softmax, CE, bobot CE,
    sampler, cara baca dan metrik TIDAK berubah. Butuh PLR nyala. Nama varian + '_plry'.
    Contoh: python metode_usulan_cls_standar.py --dataset shoppers --ratio 0.3 --seed 0 --parts val test --readout pm --plr_y --hapus_ckpt

Contoh:
    python metode_usulan_cls_standar.py --dataset shoppers --ratio 0.3 --seed 2 --parts val test --hapus_ckpt
    python metode_usulan_cls_standar.py --dataset shoppers --ratio 0.3 --seed 2 --parts val test --readout pm
    python metode_usulan_cls_standar.py --dataset shoppers --ratio 0.3 --seed 0 --quick          # uji alur cepat
"""

import os
import copy
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

GROUP = 'usulan_cls_standar'
EPS = 1e-6


# ======================================================================
# Jaringan
# ======================================================================
class PrecondMaskCE(PrecondMask):
    """Head klasifikasi untuk blok y: keluaran mentah F_x kolom y = LOGIT; denoiser blok y = softmax(logit)
    = peluang kelas = taksiran E[y_onehot {0,1} | x_t]. Kolom fitur = PrecondMask (tidak berubah)."""
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
        D = torch.cat([D[:, :self.y_cols[0]], torch.softmax(logits, dim=1), D[:, self.y_cols[-1] + 1:]], dim=1)
        return D, logits

    def forward(self, x, sigma, c=None):
        return self.forward_logits(x, sigma, c)[0]


def build_net(dim, args, device, print_model):
    if args.plr:                                                    # PLR (+ penanda c)
        if args.task == 'regression':
            y_plr = [args.y_cols[0]]                                  # regresi: y selalu lewat PLR
        else:
            y_plr = list(args.y_cols) if args.plr_y else []           # klasifikasi: blok y lewat PLR bila --plr_y
        num_idx = list(range(args.n_num)) + y_plr
        denoise_fn = MLPDiffusionPLR(dim, num_idx, args.hid_dim, args.plr_k, args.plr_dim,
                                     args.plr_sigma).to(device)
        if print_model:
            print(denoise_fn)
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


def ce_hi_loss(net, data, mask_rows, y_cols, s_min, s_max):
    """[--ce_hi] CE blok y di sigma tinggi, semua baris KONDISIONAL (fitur teramati bersih, fitur bolong & y ditebak).
    sigma ~ log-uniform[s_min, s_max]. Kembali CE per baris."""
    c = mask_rows.clone()
    c[:, y_cols] = 1.0
    lo, hi = math.log(s_min), math.log(s_max)
    sigma = torch.exp(lo + (hi - lo) * torch.rand(data.shape[0], device=data.device))
    n = torch.randn_like(data) * sigma.unsqueeze(1) * c
    _, logits = net.forward_logits(data + n, sigma, c)
    return torch.nn.functional.cross_entropy(logits, data[:, y_cols].argmax(1), reduction='none')


@torch.no_grad()
def probe_val(net, Xv, Mv, yv, y_cols, sigmas, seed):
    """[--probe_every] 1 forward kondisional per sigma pada VAL (c = mask val: fitur bolong & y). Noise dari Generator
    sendiri (seed tetap -> noise sama tiap probe, RNG training tidak tersentuh). Kembali {sigma: (CE, AUC)}."""
    was_training = net.training
    net.eval()
    g = torch.Generator(device=Xv.device)
    g.manual_seed(seed)
    out = {}
    for s in sigmas:
        sig = torch.full((Xv.shape[0],), float(s), device=Xv.device)
        n = torch.randn(Xv.shape, generator=g, device=Xv.device) * float(s) * Mv
        _, logits = net.forward_logits(Xv + n, sig, Mv)
        ce = torch.nn.functional.cross_entropy(logits, yv).item()
        P = torch.softmax(logits, 1).cpu().numpy()
        out[float(s)] = (ce, cls_metrics(P, yv.cpu().numpy())['auc'])
    if was_training:
        net.train()
    return out


def train_mstep(train_data, train_mask, dim, args, device, ckpt_iter_dir, print_model, probe_data=None,
                init_state=None):
    """M-step. Loss = (sum lambda(sigma) * w_kolom * c * (D - x)^2 [+ ce_weight * sum CE_y] [+ ce_hi * sum CE_hi])
    / sum c, early stopping dari loss tanpa bobot kolom. probe_data = (X_obs, mask, y) val untuk --probe_every.
    Kembali (net, riwayat probe)."""
    X = torch.tensor(train_data, dtype=torch.float32, device=device)
    M = torch.tensor(train_mask, dtype=torch.float32, device=device)
    net = build_net(dim, args, device, print_model)
    if init_state is not None:                                        # [--warm_start] lanjut dari iterasi sebelumnya
        net.load_state_dict(torch.load(init_state, map_location=device))
        print(f'[warm_start] M-step mulai dari {init_state}')
    ema_net = None
    if args.ema > 0:                                                  # [--ema] salinan bobot rata-rata bergerak
        ema_net = copy.deepcopy(net)
        for q in ema_net.parameters():
            q.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)
    col_w = torch.ones(dim, device=device)
    col_w[args.y_cols] = args.lam / len(args.y_cols)
    simpan_net = ema_net if ema_net is not None else net              # bobot yang disimpan & dipakai
    n_upd = 0

    pakai_probe = args.loss_y == 'ce' and args.probe_every > 0 and probe_data is not None
    if pakai_probe:
        Xv = torch.tensor(probe_data[0], dtype=torch.float32, device=device)
        Mv = torch.tensor(probe_data[1], dtype=torch.float32, device=device)
        yv = torch.tensor(probe_data[2], dtype=torch.long, device=device)
    riwayat, best_v = [], float('inf')

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
                if args.ce_hi > 0:                                    # [--ce_hi] CE di sigma tinggi
                    ce2 = ce_hi_loss(net, X[idx], M[idx], args.y_cols, args.ce_hi_min, args.ce_hi_max)
                    loss = loss + args.ce_hi * ce2.sum() / cnt
                    err_log += ce2.detach().sum().item()
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
            if ema_net is not None:                                   # peluruhan dengan pemanasan: awal training
                n_upd += 1                                            # EMA tidak terseret bobot acak awal
                dk = min(args.ema, (1 + n_upd) / (10 + n_upd))
                with torch.no_grad():
                    for q_e, q in zip(ema_net.parameters(), net.parameters()):
                        q_e.mul_(dk).add_(q.detach(), alpha=1 - dk)
            tot_err += err_log
            tot_cnt += cnt.item()
        cur = tot_err / tot_cnt
        sched.step(cur)
        if pakai_probe and (epoch % args.probe_every == 0):           # [--probe_every] diagnosa val
            pr = probe_val(simpan_net, Xv, Mv, yv, args.y_cols, args.probe_sigmas, seed=12345)
            ce_v = float(np.mean([v[0] for v in pr.values()]))
            riwayat.append({'epoch': epoch, 'loss_train': cur, 'ce_val_rata': ce_v,
                            **{f'ce_s{s:g}': v[0] for s, v in pr.items()},
                            **{f'auc_s{s:g}': v[1] for s, v in pr.items()}})
            if args.val_ckpt and ce_v < best_v:                       # [--val_ckpt] simpan model val terbaik
                best_v = ce_v
                torch.save(simpan_net.state_dict(), f'{ckpt_iter_dir}/model_val.pt')
        if cur < best:
            best, patience = cur, 0
            torch.save(simpan_net.state_dict(), f'{ckpt_iter_dir}/model.pt')
        else:
            patience += 1
            if patience == args.patience:
                print('Early stopping')
                break
        pbar.set_postfix(loss=cur)
    pbar.close()
    pilih = f'{ckpt_iter_dir}/model_val.pt' if (args.val_ckpt and os.path.exists(f'{ckpt_iter_dir}/model_val.pt')) \
        else f'{ckpt_iter_dir}/model.pt'
    net.load_state_dict(torch.load(pilih, map_location=device))
    net.eval()
    if riwayat:
        b = min(riwayat, key=lambda r: r['ce_val_rata'])
        print(f'[probe val] epoch berhenti {epoch} | CE val rata-rata terbaik {b["ce_val_rata"]:.4f} di epoch '
              f'{b["epoch"]} | akhir {riwayat[-1]["ce_val_rata"]:.4f} | checkpoint dipakai: {os.path.basename(pilih)}')
    return net, {'epoch_berhenti': epoch, 'checkpoint': os.path.basename(pilih), 'riwayat': riwayat}


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
def probs_posterior_mean(D):
    """Rekaman denoiser blok y [N ulangan, steps, n, K] = softmax(logit) tiap langkah -> peluang per langkah
    [steps, n, K] = rata-rata softmax N ulangan. Langsung peluang (jumlah 1); tanpa +0.5, clip, normalisasi."""
    return D.mean(0)


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


KRITERIA_PM = 'auc'          # diisi dari --pilih_pm di main()


def pilih_langkah(curve_val, y_val, is_reg):
    """Langkah sigma posterior mean di VAL: regresi RMSE terkecil; klasifikasi AUC terbesar (default) atau
    [--pilih_pm logloss] log-loss (CE) terkecil. AUC tidak peduli besar peluang -> bisa memilih langkah sigma besar
    yang peluangnya menciut ke prior (argmax = kelas mayoritas, F1 runtuh); log-loss menilai urutan DAN besar peluang."""
    if is_reg:
        return int(np.argmin([np.sqrt(((y_val - c) ** 2).mean()) for c in curve_val]))
    if KRITERIA_PM == 'logloss':
        ll = [-np.log(np.clip(c[np.arange(len(y_val)), y_val], 1e-12, 1.0)).mean() for c in curve_val]
        return int(np.argmin(np.nan_to_num(np.array(ll), nan=np.inf)))
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
    if not is_reg:                                                    # one-hot {-0.5, +0.5} -> {0, 1}
        yc = meta['y_cols']
        for part in data:
            data[part]['X_true'][:, yc] += 0.5
            data[part]['X_obs'][:, yc] += 0.5                         # sel y val/test tetap di-mask (ditebak)
        meta['clip_lo'] = np.array(meta['clip_lo'], dtype=float).copy()
        meta['clip_hi'] = np.array(meta['clip_hi'], dtype=float).copy()
        meta['clip_lo'][yc] += 0.5
        meta['clip_hi'][yc] += 0.5
        assert set(np.unique(data['train']['X_obs'][:, yc])) <= {0.0, 1.0}, 'blok y train harus one-hot {0, 1}'
    dim, yi = meta['dim'], meta['y_idx']
    X_tr, M_tr = data['train']['X_obs'], data['train']['mask']
    y_true = {p: (y_true_z(data[p]['df'], meta) if is_reg else y_true_index(data[p]['df'], meta)) for p in parts}
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | {args.task} | device {device} | '
          f'train {X_tr.shape}, val {data["val"]["X_obs"].shape}, test {data["test"]["X_obs"].shape}')
    print(f'Varian {args.varian} | {args.info}')

    summary = {'varian': args.varian, 'task': args.task, 'dataset': args.dataset, 'mech': args.mech,
               'ratio': args.ratio, 'seed': args.seed, 'komponen': args.info, 'args': vars(args), 'iterasi': []}
    REC = {'sampel': {p: [] for p in parts}, 'pm': {p: [] for p in parts}}   # per iterasi yang dievaluasi
    lab_s = 'sampel' if is_reg else 'akhir'                         # cara baca kedua (klasifikasi: langkah terakhir)
    S_last, iters = {}, []
    t_all = time.time()

    for it in range(args.max_iter):
        print('=' * 60)
        print(f'Iteration: {it}')
        os.makedirs(f'{ckpt_dir}/{it}', exist_ok=True)
        X_miss = X_tr.copy() if it == 0 else np.load(f'{ckpt_dir}/iter_{it}.npy')

        t0 = time.time()                                              # ---- M-step ----
        probe_data = None if is_reg else (data['val']['X_obs'], data['val']['mask'], y_true['val'])
        init_state = None
        if args.warm_start and it > 0:                                # [--warm_start] bobot yang dipakai iterasi lalu
            prev = f'{ckpt_dir}/{it - 1}'
            init_state = f'{prev}/model_val.pt' if (args.val_ckpt and os.path.exists(f'{prev}/model_val.pt')) \
                else f'{prev}/model.pt'
            if not os.path.exists(init_state):
                raise SystemExit(f'--warm_start: checkpoint iterasi sebelumnya tidak ada: {init_state}')
        net, info_m = train_mstep(X_miss, M_tr, dim, args, device, f'{ckpt_dir}/{it}', print_model=(it == 0),
                                  probe_data=probe_data, init_state=init_state)
        res = {'iterasi': it, 'waktu_train_s': time.time() - t0, 'epoch_berhenti': info_m['epoch_berhenti'],
               'checkpoint': info_m['checkpoint']}
        if info_m['riwayat']:
            res['probe'] = info_m['riwayat']

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
                    curve = probs_posterior_mean(D)                     # [steps, n, K] rata-rata softmax
                    samp = curve[-1]                                    # cara baca 'akhir' = langkah terakhir
                    np.save(f'{prefix}_{part}_pmkurva_iter{it}.npy', curve.astype(np.float32))   # pilih ulang tanpa run
                    if not args.tanpa_lintasan:                         # peluang PER LINTASAN (analisis ketidakpastian)
                        np.save(f'{prefix}_{part}_lintasan_iter{it}.npy', D.astype(np.float16))  # [N, langkah, n, K]
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
                    res[part].update({lab_s: m_s, 'pm': {**m_p, 'langkah': s_it, 'sigma': float(sig[s_it])}})
                    im = res[part]['imputasi']
                    line = (f'iteration {it} | {part} | imputasi MAE {im["mae"]:.6f} RMSE {im["rmse"]:.6f} '
                            f'ACC {im["acc"]} | {lab_s} {fmt(m_s, is_reg)} | pm (sigma {sig[s_it]:.3g}) '
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
            for r in ['pm', lab_s]:
                kunci = f'{r}_{h}'
                summary['hasil'][kunci] = {'langkah_pm': s_h, 'sigma_pm': float(sig[s_h]),
                                           'iterasi': iters if h == 'ensemble' else [iters[-1]]}
                for p in parts:
                    y = y_true[p]
                    if is_reg:
                        pt = curve_h[p][s_h] if r == 'pm' else samp_h[p].mean(0)
                        m = reg_metrics(y, pt, samp_h[p])
                    else:
                        pt = curve_h[p][s_h] if r == 'pm' else samp_h[p]       # akhir = rata-rata langkah terakhir
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
    if args.task == 'classification':
        nama += '_ce01' + (f'_w{args.ce_weight:g}' if args.ce_weight != args.lam else '')
    if args.plr_y:
        nama += '_plry'
    if args.ce_hi > 0:
        nama += f'_hi{args.ce_hi:g}' + (f'_m{args.ce_hi_min:g}' if args.ce_hi_min != 3.0 else '') + \
                (f'_x{args.ce_hi_max:g}' if args.ce_hi_max != 80.0 else '')
    if args.val_ckpt:
        nama += '_vck'
    if args.task == 'classification' and args.pilih_pm == 'logloss':
        nama += '_ll'
    if args.ema > 0:
        nama += f'_ema{args.ema:g}'
    if args.warm_start:
        nama += '_ws'
    if args.plr:
        nama += (f'_sig{args.plr_sigma:g}' if args.plr_sigma != 1.0 else '') + \
                (f'_k{args.plr_k}' if args.plr_k != 48 else '') + (f'_d{args.plr_dim}' if args.plr_dim != 24 else '')
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
    p.add_argument('--ce_weight', type=float, default=None, help='KLASIFIKASI: bobot CE per baris (default = --lam)')
    p.add_argument('--plr_y', action='store_true',
                   help='KLASIFIKASI: K kolom blok y (one-hot ber-noise) ikut lewat PLR, seperti y regresi. '
                        'Hanya sisi input; loss CE, logit, softmax, prediksi tidak berubah. Default mati.')
    p.add_argument('--ce_hi', type=float, default=0.0,
                   help='KLASIFIKASI: bobot CE tambahan di sigma tinggi (baris kondisional, y belum diketahui). 0 = mati')
    p.add_argument('--ce_hi_min', type=float, default=3.0, help='batas bawah sigma --ce_hi (log-uniform)')
    p.add_argument('--ce_hi_max', type=float, default=80.0, help='batas atas sigma --ce_hi (= sigma_max sampler)')
    p.add_argument('--probe_every', type=int, default=0,
                   help='KLASIFIKASI: probe CE/AUC val tiap E epoch (diagnosa; training tidak berubah). 0 = mati')
    p.add_argument('--probe_sigmas', type=float, nargs='+', default=[80.0, 10.0, 3.0, 1.0])
    p.add_argument('--tanpa_lintasan', action='store_true',
                   help='KLASIFIKASI: jangan simpan peluang per lintasan {prefix}_{part}_lintasan_iter{it}.npy')
    p.add_argument('--ema', type=float, default=0.0,
                   help='EMA bobot di M-step, mis. 0.999 (0 = mati). Checkpoint yang dipakai = bobot EMA')
    p.add_argument('--warm_start', action='store_true',
                   help='M-step iterasi k mulai dari bobot iterasi k-1 (Generalized EM)')
    p.add_argument('--pilih_pm', default='auc', choices=['auc', 'logloss'],
                   help='KLASIFIKASI: kriteria memilih langkah sigma pm di VAL (default auc = versi lama; logloss = CE)')
    p.add_argument('--val_ckpt', action='store_true',
                   help='KLASIFIKASI: checkpoint M-step = CE val probe terkecil (early stopping tetap dari loss train)')
    p.add_argument('--readout', default=None, choices=['pm', 'sampel', 'akhir'],
                   help="hasil utama. Regresi: pm (default) | sampel. Klasifikasi: akhir (default) | pm")
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
    is_cls = args.task == 'classification'
    args.loss_y = 'ce' if is_cls else 'mse'                           # klasifikasi SELALU softmax + CE
    args.emb_kategorik = False
    if args.readout is None:
        args.readout = 'akhir' if is_cls else 'pm'
    if args.readout not in (('akhir', 'pm') if is_cls else ('pm', 'sampel')):
        raise SystemExit(f'--readout {args.readout} tidak berlaku untuk {args.task}.')
    if is_cls and not (args.masking or args.plr):
        raise SystemExit('Klasifikasi (softmax + CE) butuh jaringan dengan penanda c (masking atau PLR nyala).')
    if args.plr_y and not is_cls:
        raise SystemExit('--plr_y hanya untuk klasifikasi (y regresi sudah selalu lewat PLR).')
    if args.plr_y and not args.plr:
        raise SystemExit('--plr_y butuh PLR nyala (jangan dipakai bersama --no_plr).')
    if not is_cls and (args.ce_hi > 0 or args.probe_every > 0 or args.val_ckpt):
        raise SystemExit('--ce_hi / --probe_every / --val_ckpt hanya untuk klasifikasi.')
    if args.ce_hi > 0 and not (0 < args.ce_hi_min < args.ce_hi_max):
        raise SystemExit('--ce_hi butuh 0 < --ce_hi_min < --ce_hi_max.')
    global KRITERIA_PM
    KRITERIA_PM = args.pilih_pm
    if args.val_ckpt and args.probe_every <= 0:
        args.probe_every = 50
    if args.hasil == 'ensemble' and args.eval_last_only:
        raise SystemExit('--hasil ensemble butuh evaluasi tiap iterasi: jangan pakai --eval_last_only.')
    if not 0 <= args.ema < 1:
        raise SystemExit('--ema harus di [0, 1), mis. 0.999 (0 = mati).')
    if args.varian == 'auto':
        args.varian = nama_varian(args)
    nyala = [n for n, on in [('masking', args.masking), ('PLR', args.plr), ('SEM', args.sem),
                             (f'lambda {args.lam:g}', args.lam != 1)] if on]
    args.info = (f'komponen: {" + ".join(nyala) or "dasar DiffPuter"} | E-step: '
                 f'{"SEM 1 sampel" if args.sem else f"rata-rata {args.num_trials} sampel"} | sampling uji: '
                 f'{"kondisional" if args.masking else "full noise (DiffPuter)"} | fitur numerik: {args.num_transform}'
                 + (f' | PLR sigma {args.plr_sigma:g}, K {args.plr_k}, d {args.plr_dim}' if args.plr else '')
                 + (' | PLR juga untuk blok y (--plr_y)' if args.plr_y else '')
                 + (f' | CE sigma tinggi bobot {args.ce_hi:g}, sigma log-uniform[{args.ce_hi_min:g}, '
                    f'{args.ce_hi_max:g}]' if args.ce_hi > 0 else '')
                 + (' | checkpoint M-step = CE val probe terkecil' if args.val_ckpt else '')
                 + (' | langkah pm dipilih dari log-loss val' if is_cls and args.pilih_pm == 'logloss' else '')
                 + (f' | y one-hot {{0,1}}, logit -> softmax -> CE (bobot {args.ce_weight:g}); prediksi rata-rata '
                    f'softmax {args.num_trials} ulangan -> argmax' if args.task == 'classification' else '')
                 + f' | hasil utama: {args.readout} x {args.hasil}')

    ckpt_dir = run(args)
    if args.hapus_ckpt:
        ck = os.path.dirname(ckpt_dir)
        if os.path.isdir(ck):
            shutil.rmtree(ck)
            print(f'[hapus_ckpt] checkpoint dihapus: {ck}')


if __name__ == '__main__':
    main()
