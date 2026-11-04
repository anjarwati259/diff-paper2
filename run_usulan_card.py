"""
METODE USULAN + CARD RESIDUAL (Han dkk., NeurIPS 2022) — file terpisah dari run_usulan_final.py.

    PLR + target-aware masking + Stochastic EM + bobot tetap lambda + residual CARD dengan f = MLP-PLR

Ide CARD: jaringan f(x_obs, m) (pretrain_f.py) menebak E[y | x_obs]; diffusion memodelkan RESIDUAL
r = y - f(x) (skala model), dikondisikan pada f(x). Prediksi y = f(x) + r.

Alur sekali jalan:
  0. f: bila f_pretrain/{ds}/{tag}/seed{k}/f_{train,val,test}.npy belum ada, pretrain_f.py dijalankan otomatis
     (grid di seed 0 bila grid_best.json belum ada -> crossfit 5 fold seed ini). Train = prediksi OUT-OF-FOLD.
  1. Matriks diffusion: [fitur (numerik + bit kategorik) | f(x) | r~], r~ = r / std_r * 0.5 (std 0.5 = sigma_data
     EDM; std_r = std residual out-of-fold train). Kolom f(x) selalu teramati (mask 0); r~ teramati di train,
     ditebak di val/test. PLR untuk fitur numerik + f(x) + r~ (bit kategorik apa adanya).
  2. EM (CARD dipakai di SETIAP langkah):
       M-step : masking p (baris kondisional: fitur teramati & f(x) bersih, fitur bolong & r~ diberi noise;
                baris full noise: semua kolom), bobot lambda pada kolom r~, early stopping loss train (sama dgn usulan)
       E-step : SEM 1 sampel, mode full noise seperti usulan; f(x) dan r~ train DIKETAHUI (ditimpa tiap langkah),
                jadi isian fitur train memakai informasi f(x) dan residual
       Test   : mode kondisional; fitur teramati & f(x) bersih, fitur bolong & r~ ditebak.
                Sampel y (skala model) = f(x) + r~ * 2 * std_r -> skala standar-log (inverse_y)
  + clip hasil generate ke rentang train +-5% (termasuk f(x) & r~).

Evaluasi: SKALA STANDAR-LOG saja (z-score log1p(y) statistik train), sama dengan run lain. Per iterasi dicetak
juga 'f saja' sebagai pembanding. Sampel y per iterasi disimpan (_samples_iter{i}.npy, skala standar-log) ->
eval_zlog.py bisa menghitung ensemble antar iterasi.

Hasil (folder baru, tidak campur):
  results_zlog/usulan_card/{ds}/{tag}/seed{k}/usulan_card_plr_mask_sem_lam10_p0.5_{result.txt, metrics.json,
      val.csv, test.csv, {part}_samples.npy, {part}_samples_iter{i}.npy}
  ckpt/usulan_card/{ds}/{tag}/seed{k}/...

Contoh:
  python run_usulan_card.py --dataset bike --ratio 0.3 --seed 0 --parts val test
  python run_usulan_card.py --dataset bike --ratio 0.3 --seed 0 --quick          # uji alur cepat
Default sama dengan usulan final: --lam 10, --p_cond 0.5, --num_trials 10, PLR K = 48, d = 24, sigma = 1, SEM.
Hanya REGRESI.
"""

import os
import sys
import json
import time
import subprocess

import numpy as np
import torch
from tqdm import tqdm

from data_loader import load_split, inverse_y, y_true_z
from run_diffputer import imputation_metrics, y_metrics, save_predictions, clip_generated
from run_usulan import build_parser, estep_train
from run_usulan_mask import PrecondMask, edm_loss_mask, run_trials_cond, add_mask_args, varian_suffix
from run_usulan_final import make_cond_y
from run_usulan_plr_mask_sem_lamfix import MLPDiffusionPLR

GROUP = 'usulan_card'
CLIP_MARGIN = 0.05


# ======================================================================
# f (pretrain_f.py): muat, atau jalankan otomatis bila belum ada
# ======================================================================
def ensure_f(args, tag):
    root = f'{args.f_dir}/{args.dataset}/{tag}'
    sd = f'{root}/seed{args.seed}'
    need = [f'{sd}/f_{p}.npy' for p in ('train', 'val', 'test')]
    py = sys.executable
    common = ['--dataset', args.dataset, '--ratio', str(args.ratio), '--mech', args.mech,
              '--out_dir', args.f_dir, '--gpu', str(args.gpu)]
    if not os.path.exists(f'{root}/grid_best.json'):
        print(f'[f] grid_best.json belum ada -> pretrain_f.py --mode grid (seed 0, pilih di val)', flush=True)
        subprocess.run([py, 'pretrain_f.py', '--mode', 'grid', '--seed', '0'] + common, check=True)
    if args.refit_f or not all(os.path.exists(f) for f in need):
        print(f'[f] f seed {args.seed} belum ada -> pretrain_f.py --mode crossfit', flush=True)
        subprocess.run([py, 'pretrain_f.py', '--mode', 'crossfit', '--seed', str(args.seed)] + common, check=True)
    f = {p: np.load(f'{sd}/f_{p}.npy').astype(np.float64) for p in ('train', 'val', 'test')}
    info = json.load(open(f'{sd}/metrics.json')) if os.path.exists(f'{sd}/metrics.json') else {}
    return f, info


# ======================================================================
# Matriks CARD: [fitur | f(x) | r~]
# ======================================================================
def build_card_data(data, meta, f):
    yi = meta['y_idx']
    y_tr = data['train']['X_true'][:, yi]                         # skala model (z/2)
    r_tr = y_tr - f['train']                                      # residual out-of-fold
    std_r = float(r_tr.std())
    out = {}
    for p, d in data.items():
        feat_obs, feat_true = d['X_obs'][:, :yi], d['X_true'][:, yi:yi] if False else d['X_true'][:, :yi]
        y = d['X_true'][:, yi]
        r_s = (y - f[p]) / std_r * 0.5
        is_tr = p == 'train'
        X_obs = np.concatenate([feat_obs, f[p][:, None], (r_s if is_tr else np.zeros_like(r_s))[:, None]], 1)
        X_true = np.concatenate([feat_true, f[p][:, None], r_s[:, None]], 1)
        M = np.concatenate([d['mask'][:, :yi], np.zeros((len(y), 1), bool),
                            np.full((len(y), 1), not is_tr)], 1)
        out[p] = {'X_obs': X_obs, 'X_true': X_true, 'mask': M}
    # batas clip: fitur dari data_loader; f & r~ = rentang train +-5%
    lo, hi = list(meta['clip_lo'][:yi]), list(meta['clip_hi'][:yi])
    for col in (f['train'], out['train']['X_true'][:, -1]):
        mn, mx = float(col.min()), float(col.max())
        lo.append(mn - CLIP_MARGIN * (mx - mn)); hi.append(mx + CLIP_MARGIN * (mx - mn))
    return out, std_r, (np.array(lo), np.array(hi))


def to_orig_layout(S, f_part, std_r, yi):
    """Sampel matriks CARD [..., yi + 2] -> tata letak data_loader [..., yi + 1] dengan kolom y skala model."""
    y = f_part + S[..., yi + 1] * 2.0 * std_r
    return np.concatenate([S[..., :yi], y[..., None]], -1)


# ======================================================================
# Jaringan & M-step
# ======================================================================
def build_net(dim, args, device, print_model):
    num_idx = list(range(args.n_num)) + [args.f_idx, args.r_idx]          # PLR: fitur numerik + f(x) + r~
    denoise_fn = MLPDiffusionPLR(dim, num_idx, args.hid_dim, args.plr_k, args.plr_dim, args.plr_sigma).to(device)
    if print_model:
        print(denoise_fn)
        print(f'[CARD] PLR untuk {len(num_idx)} kolom (fitur numerik + f(x) + r~); {len(denoise_fn.other_idx)} '
              f'bit kategorik apa adanya; parameter = {sum(p.numel() for p in denoise_fn.parameters()):,}')
    return PrecondMask(denoise_fn).to(device)


def train_mstep(train_data, train_mask, dim, args, device, ckpt_iter_dir, print_model):
    """= train_mstep run_usulan_final (masking + bobot tetap lambda), y diganti kolom r~."""
    X = torch.tensor(train_data, dtype=torch.float32, device=device)
    M = torch.tensor(train_mask, dtype=torch.float32, device=device)
    net = build_net(dim, args, device, print_model)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)
    col_w = torch.ones(dim, device=device)
    col_w[args.r_idx] = args.lam
    best, patience = float('inf'), 0
    n = X.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        perm = torch.randperm(n, device=device)
        tot_err, tot_cnt = 0.0, 0.0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            c = make_cond_y(M[idx], args.p_cond, [args.r_idx])            # f(x) bersih di baris kondisional
            err = edm_loss_mask(net, X[idx], c)
            loss = (err * col_w).sum() / c.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot_err += err.detach().sum().item(); tot_cnt += c.sum().item()
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


def point_metrics(y, p):
    return {'rmse': float(np.sqrt(((y - p) ** 2).mean())), 'mae': float(np.abs(y - p).mean()),
            'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}


# ======================================================================
# Alur EM
# ======================================================================
def main():
    p = add_mask_args(build_parser('Usulan + residual CARD (f = MLP-PLR)'))
    p.add_argument('--lam', type=float, default=10.0, help='bobot kolom residual r~')
    p.add_argument('--plr_k', type=int, default=48)
    p.add_argument('--plr_dim', type=int, default=24)
    p.add_argument('--plr_sigma', type=float, default=1.0)
    p.add_argument('--f_dir', default='f_pretrain', help='folder hasil pretrain_f.py')
    p.add_argument('--refit_f', action='store_true', help='paksa crossfit ulang f untuk seed ini')
    p.set_defaults(num_trials=10)
    args = p.parse_args()
    args.sem = True
    if args.quick:
        args.max_iter, args.max_epochs, args.patience = 2, 200, 50
        args.num_steps, args.resample, args.num_trials = 10, 2, 5
    if args.resample_test is None:
        args.resample_test = args.resample
    if args.varian == 'auto':
        args.varian = f'usulan_card_plr_mask_sem_lam{args.lam:g}_{varian_suffix(args)}' + ('_quick' if args.quick else '')

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    tag = f'{args.mech}{int(round(args.ratio * 100)):02d}'
    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech)
    if meta.get('task', 'regression') != 'regression':
        raise SystemExit('run_usulan_card.py hanya untuk regresi')

    f, f_info = ensure_f(args, tag)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cd, std_r, bounds = build_card_data(data, meta, f)
    yi = meta['y_idx']
    args.n_num, args.f_idx, args.r_idx = meta['n_num'], yi, yi + 1
    dim = yi + 2

    ckpt_dir = f'ckpt/{GROUP}/{args.dataset}/{tag}/seed{args.seed}/{args.varian}/{args.num_trials}_{args.num_steps}'
    pred_dir = f'{args.out_dir}/{GROUP}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    result_path = f'{pred_dir}/{args.varian}_result.txt'
    open(result_path, 'w').close()
    X_tr, M_tr = cd['train']['X_obs'], cd['train']['mask']
    print(f'Dataset {args.dataset} | {tag} | seed {args.seed} | device {device} | matriks CARD train {X_tr.shape} '
          f'(fitur {yi} + f(x) + r~) | std residual train (skala model) {std_r:.4f}')
    print(f'Varian {args.varian} | lambda r~ = {args.lam:g} | masking p = {args.p_cond:g} | SEM | N = {args.num_trials}')

    yz = {pt: y_true_z(data[pt]['df'], meta) for pt in args.parts}
    f_only = {pt: point_metrics(yz[pt], inverse_y(f[pt], meta)) for pt in args.parts}
    for pt in args.parts:
        m = f_only[pt]
        print(f'f saja ({pt}) → RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}  R2={m["r2"]:.4f}  (skala standar-log)')

    summary = {'varian': args.varian, 'kelompok': GROUP, 'dataset': args.dataset, 'mech': args.mech,
               'ratio': args.ratio, 'seed': args.seed, 'std_r_model': std_r, 'f_saja': f_only,
               'f_pretrain': f_info.get('cfg'), 'args': vars(args), 'iterations': []}
    t_all = time.time()
    for it in range(args.max_iter):
        print('=' * 60); print(f'Iteration: {it}')
        os.makedirs(f'{ckpt_dir}/{it}', exist_ok=True)
        X_miss = X_tr.copy() if it == 0 else np.load(f'{ckpt_dir}/iter_{it}.npy')

        t0 = time.time()
        net = train_mstep(X_miss, M_tr, dim, args, device, f'{ckpt_dir}/{it}', print_model=(it == 0))
        it_res = {'iteration': it, 'train_time_s': time.time() - t0}
        print(f'Training time: {it_res["train_time_s"]:.1f}s')

        # E-step (SEM, full noise): f(x) & r~ train diketahui -> isian fitur memakai informasi CARD
        if M_tr.any():
            t0 = time.time()
            rec = estep_train(net, X_tr, M_tr, args, device, bounds)
            np.save(f'{ckpt_dir}/iter_{it + 1}.npy', rec)
            rec_o = to_orig_layout(rec, f['train'], std_r, yi)
            mae, rmse, acc, acc_n = imputation_metrics(rec_o, data['train']['X_true'], data['train']['mask_feat'], meta)
            print(f'In-sample  → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}  ({time.time() - t0:.0f}s)')
            it_res['in_sample'] = {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n}
        else:
            np.save(f'{ckpt_dir}/iter_{it + 1}.npy', X_tr)

        last = it == args.max_iter - 1
        if last or not args.eval_last_only:
            for pt in args.parts:
                t0 = time.time()
                S = run_trials_cond(net, cd[pt]['X_obs'], cd[pt]['mask'], args, device,
                                    f'Out-of-sample kondisional ({pt})', bounds)          # [N, n, yi + 2]
                So = to_orig_layout(S, f[pt], std_r, yi)                                     # [N, n, yi + 1]
                mae, rmse, acc, acc_n = imputation_metrics(So.mean(0), data[pt]['X_true'], data[pt]['mask_feat'], meta)
                y_samples = inverse_y(So[:, :, yi], meta)                                    # skala standar-log
                np.save(f'{pred_dir}/{args.varian}_{pt}_samples_iter{it}.npy', y_samples.T.astype(np.float32))
                ym = y_metrics(yz[pt], y_samples)
                t_out = time.time() - t0
                print(f'Out-of-sample ({pt}) → MAE={mae:.6f}  RMSE={rmse:.6f}  ACC={acc}')
                print(f'Prediksi y ({pt}) → RMSE={ym["rmse"]:.4f}  MAE={ym["mae"]:.4f}  R2={ym["r2"]:.4f}  '
                      f'CRPS={ym["crps"]:.4f}  cov p5-95={ym["coverage_p5_p95"]:.3f}  cov min-max={ym["coverage_minmax"]:.3f} '
                      f'(ideal {ym["coverage_minmax_ideal"]:.3f}) | f saja RMSE {f_only[pt]["rmse"]:.4f}')
                it_res[pt] = {'imputasi': {'mae': mae, 'rmse': rmse, 'acc': acc, 'acc_terdekat': acc_n},
                              'y': ym, 'time_s': t_out}
                if last:
                    save_predictions(f'{pred_dir}/{args.varian}', pt, data[pt], So, y_samples, meta, args)
                with open(result_path, 'a') as fh:
                    fh.write(f'iteration {it} | {pt} | imputasi MAE {mae:.6f} RMSE {rmse:.6f} ACC {acc} | '
                             f'y RMSE {ym["rmse"]:.4f} MAE {ym["mae"]:.4f} R2 {ym["r2"]:.4f} CRPS {ym["crps"]:.4f} '
                             f'covmm {ym["coverage_minmax"]:.3f} | f saja RMSE {f_only[pt]["rmse"]:.4f} | '
                             f'waktu test {t_out:.0f}s\n')
        summary['iterations'].append(it_res)
        with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as fh:
            json.dump(summary, fh, indent=2, default=str)

    summary['total_time_s'] = time.time() - t_all
    with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print('=' * 60)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{args.varian}_*  | checkpoint: {ckpt_dir}')


if __name__ == '__main__':
    main()