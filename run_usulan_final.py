"""
METODE USULAN FINAL — satu skrip untuk target REGRESI dan KLASIFIKASI.

    PLR + target-aware masking + Stochastic EM + bobot tetap lambda = 10   (satu model diffusion)

Komponen (sama untuk kedua tugas):
  1. Bobot tetap lambda pada y (Xin dkk. & Kurin dkk., NeurIPS 2022). Total bobot y = lambda:
       regresi      : 1 kolom y, bobot lambda                      (identik dengan versi lama)
       klasifikasi  : K kolom one-hot, bobot lambda / K per kolom  (total tetap lambda, tidak
                      membesar dengan jumlah kelas)
  2. Target-aware masking p = 0.5: baris kondisional (fitur teramati bersih, c = 0) vs full noise;
     loss hanya di sel c = 1. Semua kolom y ikut ditebak (c = 1).
  3. Stochastic EM: E-step train memakai 1 sampel acak (Celeux & Diebolt 1985).
  4. Embedding periodik PLR (Gorishniy dkk., NeurIPS 2022) untuk kolom NUMERIK:
       regresi      : fitur numerik + y
       klasifikasi  : fitur numerik saja; blok one-hot y masuk apa adanya (seperti bit kategorik)
  + clip hasil generate ke rentang train +-5% (klasifikasi: blok y ke +-0.55).

Perbedaan per tugas (--task, default 'auto' = dibaca dari datasets/Info/{dataset}.json "task_type"):
  regression     : y = log1p (bila log_target) + z-score; prediksi = rata-rata N sampel, interval =
                   persentil; metrik RMSE, MAE, R2, CRPS, coverage. Alur = run_usulan_mask.run_mask, jadi
                   hasilnya IDENTIK dengan run_usulan_plr_mask_sem_lamfix.py (seed & argumen sama).
  classification : y = one-hot +-0.5 (CARD, Han dkk. NeurIPS 2022), tidak distandardisasi.
                   PROBABILITAS = POSTERIOR MEAN (--readout pm, default): selama sampling, output denoiser
                   D(x_t; sigma) = E[x_0 | x_t] blok y direkam tiap langkah; rata-rata atas N lintasan =
                   E[y | x_obs] = p(y = k | x_obs) - 0.5 (kontinu & terkalibrasi). Langkah sigma dipilih di VAL
                   (--pm_select auc). Sampel akhir tetap untuk ketidakpastian (agreement) & pembanding
                   (--readout sampel = cara lama: rata-rata N sampel). Sampel identik dengan versi sebelumnya.
                   Ambang (biner): dipilih di VAL memaksimalkan F1 macro (--thr_objective), metrik *_thr.
                   Kelas utama = argmax (ambang 0.5). Temperature scaling (T dari val, objektif Brier) disimpan
                   di metrics.json (y_ts). Metrik utama: AUC, F1 (macro/weighted, di 0.5 dan di ambang val), Acc.
                   'val' otomatis ikut dievaluasi. Pemilihan lambda & K: AUC val.
                   Kurva per sigma: _pm_kurva.csv; probabilitas per iterasi: _{part}_prob_iter{i}.npy.

Hasil (folder baru, tidak menimpa hasil lama):
  results/usulan_final/{ds}/{tag}/seed{k}/usulan_plr_mask_sem_lam10_p0.5_{result.txt, metrics.json, ...}
  klasifikasi juga: _{part}.csv (prob per kelas), _{part}_ysamples_iter{i}.npy [n, N, K]
  ckpt/usulan_final/...

Contoh:
  python run_usulan_final.py --dataset shoppers --ratio 0.3 --seed 0 --quick              # uji alur
  python run_usulan_final.py --dataset shoppers --ratio 0.3 --seed 0 --task classification --parts val test
  python run_usulan_final.py --dataset bike --ratio 0.3 --seed 0 --task regression --parts val test
Default: --lam 10, --num_trials 10 (keputusan N = 10), --p_cond 0.5, PLR K = 48, d = 24, sigma = 1.
Transformasi fitur numerik: --num_transform none|quantile|yeojohnson (default none = hasil lama; quantile
  -> varian berakhiran _qt). Fit di nilai teramati train untuk SEMUA fitur numerik; metrik imputasi & CSV tetap
  di skala asli (data_loader.model_to_eval).
"""

import os
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_loader import load_split, clip_bounds, inverse_transform, y_true_index, model_to_eval
from run_diffputer import imputation_metrics, clip_generated, SIGMA_MIN, SIGMA_MAX, RHO, S_CHURN, S_MIN, S_MAX, S_NOISE
from run_usulan import build_parser, estep_train
from run_usulan_mask import PrecondMask, edm_loss_mask, run_trials_cond, add_mask_args, run_mask, varian_suffix
from run_usulan_plr_mask_sem_lamfix import MLPDiffusionPLR
import metrics_cls as mc

GROUP = 'usulan_final'


# ======================================================================
# Penanda c, jaringan, M-step (berlaku untuk kedua tugas)
# ======================================================================
def make_cond_y(mask_rows, p, y_cols):
    """= make_cond (run_usulan_mask.py), tetapi y bisa beberapa kolom. Regresi (y_cols = [y_idx])
    menghasilkan c dan pemakaian RNG yang persis sama."""
    B, d = mask_rows.shape
    c_cond = mask_rows.clone()
    c_cond[:, y_cols] = 1.0
    is_cond = (torch.rand(B, 1, device=mask_rows.device) < p).float()
    return is_cond * c_cond + (1 - is_cond) * torch.ones_like(c_cond)


def build_net(dim, args, device, print_model):
    """PLR untuk fitur numerik (+ y bila regresi). Regresi: sama dengan build_net_plr lama."""
    num_idx = list(range(args.n_num)) + ([args.y_cols[0]] if args.task == 'regression' else [])
    denoise_fn = MLPDiffusionPLR(dim, num_idx, args.hid_dim, args.plr_k, args.plr_dim,
                                 args.plr_sigma).to(device)
    if print_model:
        print(denoise_fn)
        print(f'[{args.task}] PLR untuk {len(num_idx)} kolom numerik'
              f'{" (termasuk y)" if args.task == "regression" else ""}; '
              f'{len(denoise_fn.other_idx)} kolom apa adanya (bit kategorik'
              f'{" + " + str(len(args.y_cols)) + " kolom one-hot y" if args.task == "classification" else ""}); '
              f'parameter = {sum(p.numel() for p in denoise_fn.parameters()):,}')
    return PrecondMask(denoise_fn).to(device)


def train_mstep(train_data, train_mask, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    """M-step: masking + bobot tetap (total lambda pada y). Kembali (net, None)."""
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
            c = make_cond_y(M[idx], args.p_cond, args.y_cols)
            err = edm_loss_mask(net, X[idx], c)
            loss = (err * col_w).sum() / c.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_err += err.detach().sum().item()
            tot_cnt += c.sum().item()
        cur = tot_err / tot_cnt                                        # loss TANPA bobot
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
# Klasifikasi: sampling kondisional + rekam D(x_t; sigma) blok y (posterior mean)
# ======================================================================
@torch.no_grad()
def impute_cond_record(net, x, mask, y_cols, num_steps, resample):
    """= impute_cond (run_usulan_mask.py), urutan acak SAMA PERSIS (sampel identik, teruji), ditambah
    rekaman output denoiser blok y di resample terakhir tiap langkah: [steps, n, K]. Kembali juga sigma (t_hat)."""
    device = x.device
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min, sigma_max = max(SIGMA_MIN, net.sigma_min), min(SIGMA_MAX, net.sigma_max)
    t_steps = (sigma_max ** (1 / RHO) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
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
    sig = (t_steps[:-1] * (1 + gamma)).cpu().numpy()
    return x_t, np.stack(rec, 0), sig


@torch.no_grad()
def run_trials_cond_record(net, X_obs, mask, y_cols, args, device, desc, bounds=None):
    """= run_trials_cond, ditambah rekaman denoiser. Kembali S [N, n, d] (sudah di-clip), D [N, steps, n, K], sigma."""
    recs, Ds, sig = [], [], None
    for _ in tqdm(range(args.num_trials), desc=desc):
        out = np.empty_like(X_obs, dtype=np.float32)
        outD = []
        for s in range(0, X_obs.shape[0], args.chunk_rows):
            xb = torch.tensor(X_obs[s:s + args.chunk_rows], dtype=torch.float32, device=device)
            mb = torch.tensor(mask[s:s + args.chunk_rows], dtype=torch.float32, device=device)
            rec, D, sig = impute_cond_record(net, xb, mb, y_cols, args.num_steps, args.resample_test)
            out[s:s + args.chunk_rows] = (rec * mb + xb * (1 - mb)).cpu().numpy()
            outD.append(D)
        recs.append(out)
        Ds.append(np.concatenate(outD, axis=1))
    return clip_generated(np.stack(recs, axis=0), mask, bounds, desc), np.stack(Ds, 0), sig


def probs_posterior_mean(D):
    """D [N, steps, n, K] -> probabilitas per langkah [steps, n, K]: E[y | x_obs] = rata-rata D atas lintasan,
    digeser +0.5, clip, normalisasi (Rao-Blackwellisasi; EDM: D(x_t; sigma) = E[x_0 | x_t])."""
    q = np.clip(D.mean(0) + 0.5, mc.EPS, 1.0)
    return q / q.sum(2, keepdims=True)


def select_step(PM_val, y_val, how='auc'):
    """Pilih langkah sigma di VAL: AUC terbesar (default) atau Brier terkecil."""
    scores = []
    for s in range(PM_val.shape[0]):
        m = mc.cls_metrics(PM_val[s], y_val)
        scores.append(m['auc'] if how == 'auc' else -m['brier'])
    scores = np.nan_to_num(np.array(scores), nan=-np.inf)
    return int(np.argmax(scores))


# ======================================================================
# Klasifikasi: simpan prediksi
# ======================================================================
def save_predictions_cls(path_prefix, part, d, S, P, q, agree, meta, args, thr=None):
    """CSV: id, fitur asli (NaN jika di-mask), {fitur}_isian, y_asli, y_pred (argmax), y_pred_ambang (biner,
    ambang dari val), p_{kelas} (cara baca utama), p_sampel_{kelas} (rata-rata sampel), agreement, entropi."""
    df = d['df']
    feat_cols = meta['num_cols'] + meta['cat_cols']
    idx_path = f"datasets/{args.dataset}/splits/seed{args.seed}/{part}_idx.npy"
    out = pd.DataFrame({'id': np.load(idx_path) if os.path.exists(idx_path) else np.arange(len(df))})
    shown = df[feat_cols].mask(d['mask_feat'])
    imputed = inverse_transform(model_to_eval(S.mean(0), meta), meta)
    for j, c in enumerate(feat_cols):
        out[c] = shown[c].values
        if d['mask_feat'][:, j].any():
            out[f'{c}_isian'] = np.where(d['mask_feat'][:, j], imputed[c].values, None)
    classes = meta['classes']
    out['y_asli'] = df[meta['target']].values
    out['y_pred'] = np.array(classes, dtype=object)[P.argmax(1)]
    if thr is not None:
        out['y_pred_ambang'] = np.array(classes, dtype=object)[mc.pred_at(P, thr)]
    for k, cl in enumerate(classes):
        out[f'p_{cl}'] = P[:, k]
    for k, cl in enumerate(classes):
        out[f'p_sampel_{cl}'] = q[:, k]
    out['agreement'] = agree
    out['entropi'] = -(P * np.log(np.clip(P, mc.EPS, 1.0))).sum(1)
    out.to_csv(f'{path_prefix}_{part}.csv', index=False)


# ======================================================================
# Klasifikasi: alur EM (sama dengan run_mask; beda di pembacaan y & metrik)
# ======================================================================
def run_cls(args, train_fn, info, group):
    if args.quick:
        args.max_iter, args.max_epochs, args.patience = 2, 200, 50
        args.num_steps, args.resample, args.num_trials = 10, 2, 5
        args.varian = args.varian + '_quick'
    if args.resample_test is None:
        args.resample_test = args.resample
    if 'val' not in args.parts:
        print("[info] klasifikasi: 'val' ditambahkan ke --parts (dibutuhkan untuk memilih sigma, ambang, T)")
    parts = ['val'] + [p for p in args.parts if p != 'val']          # val dulu -> T untuk test
    args.parts = parts

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f'{args.mech}{int(round(args.ratio * 100)):02d}'
    ckpt_dir = f'ckpt/{group}/{args.dataset}/{tag}/seed{args.seed}/{args.varian}/{args.num_trials}_{args.num_steps}'
    pred_dir = f'{args.out_dir}/{group}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    result_path = f'{pred_dir}/{args.varian}_result.txt'
    open(result_path, 'w').close()
    if os.path.exists(f'{pred_dir}/{args.varian}_pm_kurva.csv'):
        os.remove(f'{pred_dir}/{args.varian}_pm_kurva.csv')

    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech, num_transform=args.num_transform)
    dim, yc = meta['dim'], meta['y_cols']
    X_tr, M_tr = data['train']['X_obs'], data['train']['mask']
    y_true = {p: y_true_index(data[p]['df'], meta) for p in parts}
    prior = np.bincount(y_true_index(data['train']['df'], meta), minlength=len(yc)) / len(X_tr)
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | device {device} | '
          f'train {X_tr.shape}, val {data["val"]["X_obs"].shape}, test {data["test"]["X_obs"].shape}')
    print(f'KLASIFIKASI: {len(yc)} kelas {meta["classes"]} | proporsi kelas train {np.round(prior, 4).tolist()}')
    print(f'Varian {args.varian} | masking p={args.p_cond:g}, resample test={args.resample_test} | {info}'
          f' | E-step: {"SEM (1 sampel)" if getattr(args, "sem", False) else "rata-rata trial (asli)"}')

    summary = {'varian': args.varian, 'kelompok': group, 'task': 'classification', 'dataset': args.dataset,
               'mech': args.mech, 'ratio': args.ratio, 'seed': args.seed, 'classes': meta['classes'],
               'prior_train': prior.tolist(), 'args': vars(args), 'iterations': []}
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

        # ---- M-step ----
        t0 = time.time()
        net, _ = train_fn(X_miss, M_tr, dim, meta['y_idx'], args, device, f'{ckpt_dir}/{iteration}',
                          print_model=(iteration == 0))
        train_time = time.time() - t0
        print(f'Training time: {train_time:.4f}s')
        it_res = {'iteration': iteration, 'train_time_s': train_time}

        # ---- E-step (SEM, mode full noise seperti asli) ----
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

        # ---- out-of-sample: MODE KONDISIONAL (+ rekam denoiser untuk posterior mean) ----
        last = iteration == args.max_iter - 1
        if last or not args.eval_last_only:
            sel, T, thr = None, None, None
            kurva = []
            for part in parts:
                d = data[part]
                t0 = time.time()
                S, Dy, sig = run_trials_cond_record(net, d['X_obs'], d['mask'], yc, args, device,
                                                    f'Out-of-sample imputation kondisional ({part})', clip_bounds(meta))
                t_out = time.time() - t0
                print(f'Out-of-sample imputation time ({part}): {t_out:.2f}s')
                mae, rmse, acc, acc_n = imputation_metrics(S.mean(0), d['X_true'], d['mask_feat'], meta)

                Y = S[:, :, yc]                                    # [N, n, K]
                np.save(f'{pred_dir}/{args.varian}_{part}_ysamples_iter{iteration}.npy',
                        Y.transpose(1, 0, 2).astype(np.float32))  # [n, N, K]
                q = mc.probs_from_samples(Y)                       # cara baca lama (rata-rata sampel)
                PM = probs_posterior_mean(Dy)                      # [steps, n, K]
                if part == 'val':                                  # semua pilihan dibuat di VAL
                    sel = select_step(PM, y_true[part], args.pm_select)
                    P_val = PM[sel] if args.readout == 'pm' else q
                    T = mc.fit_temperature(P_val, y_true[part], objective='brier')
                    thr = mc.fit_threshold(P_val, y_true[part], objective=args.thr_objective)
                P = PM[sel] if args.readout == 'pm' else q
                np.save(f'{pred_dir}/{args.varian}_{part}_prob_iter{iteration}.npy', P.astype(np.float32))
                for s_ in range(PM.shape[0]):
                    mk = mc.cls_metrics(PM[s_], y_true[part])
                    kurva.append({'iterasi': iteration, 'part': part, 'langkah': s_, 'sigma': float(sig[s_]),
                                  'auc': mk['auc'], 'brier': mk['brier'], 'nll': mk['nll'], 'acc': mk['acc']})

                m_main = mc.cls_metrics(P, y_true[part], thr)
                m_samp = mc.cls_metrics(q, y_true[part], mc.fit_threshold(q, y_true[part]) if part == 'val' else None)
                agree = mc.vote_agreement(Y, P.argmax(1))
                m_main['agreement'] = float(agree.mean())
                m_ts = mc.cls_metrics(mc.apply_temperature(P, T), y_true[part], thr)

                baca = (f'posterior mean (langkah {sel}, sigma {sig[sel]:.3g}, dipilih {args.pm_select} val)'
                        if args.readout == 'pm' else 'rata-rata sampel')
                note = ' [ambang dipilih di val ini]' if part == 'val' else ''
                print(f'Out-of-sample ({part}) → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}  ACC_terdekat={acc_n}')
                print(f'Prediksi y ({part}, {baca}) → {mc.fmt(m_main)}  agreement={m_main["agreement"]:.3f}{note}')
                print(f'   pembanding cara baca sampel ({part}) → Acc={m_samp["acc"]:.4f}  F1(macro)={m_samp["f1_macro"]:.4f}  '
                      f'AUC={m_samp["auc"]:.4f}  Brier={m_samp["brier"]:.4f}')
                it_res[part] = {'imputasi': {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n},
                                'readout': args.readout, 'pm_langkah': sel, 'pm_sigma': float(sig[sel]),
                                'ambang': thr, 'T_brier': T, 'y': m_main, 'y_ts': m_ts, 'y_sampel': m_samp,
                                'time_s': t_out}
                if last:
                    save_predictions_cls(f'{pred_dir}/{args.varian}', part, d, S, P, q, agree, meta, args, thr)
                thr_txt = (f' | ambang {thr:.3f}: Acc {m_main["acc_thr"]:.4f} F1(macro) {m_main["f1_macro_thr"]:.4f} '
                           f'F1(w) {m_main["f1_weighted_thr"]:.4f}') if 'thr' in m_main else ''
                with open(result_path, 'a') as f:
                    f.write(f'iteration {iteration} | {part} | imputasi MAE {mae:.6f} RMSE {rmse:.6f} ACC {acc} '
                            f'ACC_terdekat {acc_n} | y {args.readout} (sigma {sig[sel]:.3g}) '
                            f'Acc {m_main["acc"]:.4f} F1(macro) {m_main["f1_macro"]:.4f} F1(w) {m_main["f1_weighted"]:.4f} '
                            f'AUC {m_main["auc"]:.4f} AUC(w) {m_main["auc_weighted"]:.4f} Brier {m_main["brier"]:.4f} '
                            f'NLL {m_main["nll"]:.4f} ECE {m_main["ece"]:.4f}{thr_txt} | '
                            f'sampel AUC {m_samp["auc"]:.4f} Brier {m_samp["brier"]:.4f} | waktu test {t_out:.0f}s\n')
            kpath = f'{pred_dir}/{args.varian}_pm_kurva.csv'
            pd.DataFrame(kurva).to_csv(kpath, mode='a', header=not os.path.exists(kpath) or iteration == 0,
                                       index=False)

        summary['iterations'].append(it_res)
        with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

    summary['total_time_s'] = time.time() - start_all
    with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('=' * 60)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{args.varian}_*  | checkpoint: {ckpt_dir}')


# ======================================================================
def main():
    p = add_mask_args(build_parser('Metode usulan final (regresi & klasifikasi)'))
    p.add_argument('--task', default='auto', choices=['auto', 'regression', 'classification'],
                   help="jenis target; 'auto' = task_type di datasets/Info/{dataset}.json")
    p.add_argument('--lam', type=float, default=10.0, help='bobot TOTAL kolom y (klasifikasi: lam/K per kolom)')
    p.add_argument('--plr_k', type=int, default=48, help='jumlah frekuensi per kolom')
    p.add_argument('--plr_dim', type=int, default=24, help='dimensi embedding per kolom')
    p.add_argument('--plr_sigma', type=float, default=1.0, help='skala inisialisasi frekuensi')
    p.add_argument('--readout', default='pm', choices=['pm', 'sampel'],
                   help="klasifikasi: 'pm' = posterior mean denoiser (utama), 'sampel' = rata-rata sampel (lama)")
    p.add_argument('--pm_select', default='auc', choices=['auc', 'brier'],
                   help='klasifikasi: kriteria memilih langkah sigma posterior mean di val')
    p.add_argument('--thr_objective', default='f1_macro', choices=['f1_macro', 'f1_weighted'],
                   help='klasifikasi biner: F1 yang dimaksimalkan saat memilih ambang di val')
    p.add_argument('--num_transform', default='none', choices=['none', 'quantile', 'yeojohnson'],
                   help="transformasi SEMUA fitur numerik (fit di nilai teramati train); 'none' = hasil lama")
    p.set_defaults(num_trials=10)                                  # keputusan N = 10
    args = p.parse_args()
    args.sem = True

    _, meta = load_split(args.dataset, args.seed, args.ratio, args.mech)   # baca struktur kolom saja
    info_task = meta['task']
    if args.task == 'auto':
        args.task = info_task
    elif args.task != info_task:
        raise SystemExit(f'--task {args.task} tidak cocok dengan task_type "{info_task}" di '
                         f'datasets/Info/{args.dataset}.json. Ubah task_type di Info lalu jalankan ulang '
                         f'(blok y dibentuk saat memuat data).')
    args.n_num = meta['n_num']
    args.y_cols = meta['y_cols']
    if args.varian == 'auto':
        args.varian = f'usulan_plr_mask_sem_lam{args.lam:g}_{varian_suffix(args)}'
        if args.num_transform != 'none':
            args.varian += {'quantile': '_qt', 'yeojohnson': '_yj'}[args.num_transform]
    info = (f'tugas = {args.task} | fitur numerik: {args.num_transform} | PLR (K={args.plr_k}, d={args.plr_dim}, sigma={args.plr_sigma:g}) | '
            f'bobot total lambda_y = {args.lam:g}'
            + (f' ({args.lam / len(args.y_cols):g} x {len(args.y_cols)} kolom one-hot)'
               if args.task == 'classification' else ''))

    if args.task == 'regression':
        run_mask(args, train_mstep, info, group=GROUP)            # alur regresi lama, apa adanya
    else:
        run_cls(args, train_mstep, info, group=GROUP)


if __name__ == '__main__':
    main()