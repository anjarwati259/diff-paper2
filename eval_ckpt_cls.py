"""
EVALUASI ULANG checkpoint metode usulan (klasifikasi) dengan beberapa cara MEMBACA PROBABILITAS.
Tidak ada training ulang: model tiap iterasi EM dimuat dari ckpt/, lalu sampling kondisional dijalankan
lagi (val & test) sambil MEREKAM output denoiser D(x_t; sigma) blok y di setiap langkah sampling.

Cara membaca probabilitas yang dibandingkan:
  1. sampel        : rata-rata N sampel akhir + 0.5, normalisasi (= cara run_usulan_final.py sekarang)
  2. laplace       : suara argmax tiap sampel, (jumlah + 1) / (N + K)
  3. card_eq10     : CARD Eq. 10 (Han dkk. 2022): per sampel p_k = softmax(-(v_k - 1)^2 / tau), v = y + 0.5,
                     lalu dirata-rata atas N sampel; tau dipilih di val (minimasi NLL)
  4. pm_sigma      : POSTERIOR MEAN dari denoiser. D(x_t; sigma) = E[x_0 | x_t] (EDM), dan rata-rata atas lintasan
                     E_{x_t}[D(x_t)] = E[x_0 | x_obs] = (p(y = k | x_obs) - 0.5) untuk blok one-hot.
                     Output denoiser kontinu (tidak "terkunci" di sudut one-hot seperti sampel akhir), jadi
                     probabilitasnya halus. Langkah sigma dipilih di val (AUC terbesar, --pm_select); kurva disimpan.
                     (Rao-Blackwellisasi; ide sama dengan prediksi titik Tweedie di catatan regresi.)
Tiap cara dilaporkan mentah dan dengan temperature scaling yang T-nya dipilih di val dengan BRIER (bukan NLL;
NLL membuat T terlalu besar untuk probabilitas kasar). Kolom f1_macro_thr = F1 macro di ambang yang dipilih di val.

Pemakaian (argumen menunjuk run yang sudah selesai; setting dibaca dari {varian}_metrics.json):
  python eval_ckpt_cls.py --dataset shoppers --ratio 0.3 --seed 0
  python eval_ckpt_cls.py --dataset shoppers --ratio 0.3 --seed 0 --iters 2 3 --num_trials 10
Hasil: results/usulan_final/{ds}/{tag}/seed{k}/{varian}_evalbaca.txt  (+ _evalbaca.json, _evalbaca_kurva.csv)
"""

import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_loader import load_split, clip_bounds, y_true_index
from run_usulan_final import build_net, GROUP, run_trials_cond_record, probs_posterior_mean, select_step
import metrics_cls as mc


def run_part(net, d, y_cols, a, device, bounds, desc):
    """N trial (sampler perekam yang sama dengan run_usulan_final.py): Y akhir [N, n, K], D [N, steps, n, K]."""
    S, D, sig = run_trials_cond_record(net, d['X_obs'], d['mask'], y_cols, a, device, desc, bounds)
    return S[:, :, y_cols], D, sig


# ======================================================================
# Cara membaca probabilitas
# ======================================================================
def p_laplace(Y):
    N, n, K = Y.shape
    votes = np.stack([(Y.argmax(2) == k).sum(0) for k in range(K)], 1)
    return (votes + 1) / (N + K)


def p_card(Y, tau):
    v = Y + 0.5
    z = -((v - 1.0) ** 2) / tau
    z -= z.max(2, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(2, keepdims=True)).mean(0)


def fit_tau(Y, y, lo=1e-3, hi=10.0):
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: mc.nll(p_card(Y, np.exp(lt)), y), bounds=(np.log(lo), np.log(hi)),
                        method='bounded')
    return float(np.exp(r.x))


def p_from_mean(M):
    q = np.clip(M + 0.5, mc.EPS, 1.0)
    return q / q.sum(1, keepdims=True)


# ======================================================================
def main():
    p = argparse.ArgumentParser(description='Evaluasi ulang checkpoint klasifikasi: cara membaca probabilitas')
    p.add_argument('--dataset', required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mech', default='MAR')
    p.add_argument('--varian', default='usulan_plr_mask_sem_lam10_p0.5')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--iters', type=int, nargs='*', default=None, help='iterasi EM (default: semua yang ada)')
    p.add_argument('--num_trials', type=int, default=None, help='N sampel (default: sama dengan run)')
    p.add_argument('--chunk_rows', type=int, default=None)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--eval_seed', type=int, default=123, help='seed sampling evaluasi ulang')
    p.add_argument('--pm_select', default='auc', choices=['auc', 'brier'], help='kriteria memilih sigma di val')
    p.add_argument('--thr_objective', default='f1_macro', choices=['f1_macro', 'f1_weighted'])
    cli = p.parse_args()

    tag = f'{cli.mech}{int(round(cli.ratio * 100)):02d}'
    pred_dir = f'{cli.out_dir}/{GROUP}/{cli.dataset}/{tag}/seed{cli.seed}'
    mpath = f'{pred_dir}/{cli.varian}_metrics.json'
    with open(mpath) as f:
        run_summary = json.load(f)
    a = SimpleNamespace(**run_summary['args'])
    if a.task != 'classification':
        raise SystemExit('skrip ini untuk klasifikasi')
    if cli.num_trials:
        a.num_trials = cli.num_trials
    if cli.chunk_rows:
        a.chunk_rows = cli.chunk_rows
    ckpt_dir = f'ckpt/{GROUP}/{cli.dataset}/{tag}/seed{cli.seed}/{a.varian}/{run_summary["args"]["num_trials"]}_{a.num_steps}'
    iters = cli.iters if cli.iters else sorted(int(x) for x in os.listdir(ckpt_dir)
                                               if x.isdigit() and os.path.exists(f'{ckpt_dir}/{x}/model.pt'))
    device = f'cuda:{cli.gpu}' if torch.cuda.is_available() and cli.gpu >= 0 else 'cpu'

    data, meta = load_split(cli.dataset, cli.seed, cli.ratio, cli.mech,
                            num_transform=getattr(a, 'num_transform', 'none'))
    yc = meta['y_cols']
    assert list(yc) == list(a.y_cols), 'struktur kolom y berbeda dengan run'
    y = {pt: y_true_index(data[pt]['df'], meta) for pt in ('val', 'test')}
    print(f'{cli.dataset} {tag} seed{cli.seed} | varian {a.varian} | iterasi {iters} | N={a.num_trials}, '
          f'langkah={a.num_steps}, resample={a.resample_test} | device {device}')

    out_txt = f'{pred_dir}/{a.varian}_evalbaca.txt'
    open(out_txt, 'w').close()
    all_res, kurva = {}, []
    keys = ['acc', 'f1_macro', 'f1_weighted', 'auc', 'f1_macro_thr', 'thr', 'brier', 'nll']

    for it in iters:
        torch.manual_seed(cli.eval_seed + it)
        np.random.seed(cli.eval_seed + it)
        net = build_net(meta['dim'], a, device, False)
        net.load_state_dict(torch.load(f'{ckpt_dir}/{it}/model.pt', map_location=device))
        net.eval()
        Y, D, sig = {}, {}, None
        for pt in ('val', 'test'):
            Y[pt], D[pt], sig = run_part(net, data[pt], yc, a, device, clip_bounds(meta), f'iter {it} {pt}')

        # ---- probabilitas per cara baca (pilihan parameter di val) ----
        tau = fit_tau(Y['val'], y['val'])
        P = {'sampel': {pt: mc.probs_from_samples(Y[pt]) for pt in Y},
             'laplace': {pt: p_laplace(Y[pt]) for pt in Y},
             f'card_eq10(tau={tau:.3g})': {pt: p_card(Y[pt], tau) for pt in Y}}
        # posterior mean per langkah: kurva di val & test, langkah dipilih di val (--pm_select, default AUC)
        PM = {pt: probs_posterior_mean(D[pt]) for pt in D}
        for st in range(PM['val'].shape[0]):
            row = {'iter': it, 'langkah': st, 'sigma': float(sig[st])}
            for pt in ('val', 'test'):
                m = mc.cls_metrics(PM[pt][st], y[pt])
                row.update({f'{pt}_{k}': m[k] for k in ('acc', 'auc', 'nll', 'brier', 'ece')})
            kurva.append(row)
        best_s = select_step(PM['val'], y['val'], cli.pm_select)
        P[f'pm_sigma(s={best_s},sigma={sig[best_s]:.3g})'] = {pt: PM[pt][best_s] for pt in PM}

        res = {}
        lines = [f'=== iterasi {it} | N={a.num_trials} | metrik test (pemilihan tau/sigma/T di val) ===',
                 f'{"cara baca":38s} {"TS":>5s} ' + ' '.join(f'{k:>11s}' for k in keys) + f' {"T":>6s}']
        for name, Pp in P.items():
            T = mc.fit_temperature(Pp['val'], y['val'], objective='brier')
            for ts in (False, True):
                Pt = mc.apply_temperature(Pp['test'], T) if ts else Pp['test']
                Pv = mc.apply_temperature(Pp['val'], T) if ts else Pp['val']
                thr = mc.fit_threshold(Pv, y['val'], objective=cli.thr_objective)
                mt, mv = mc.cls_metrics(Pt, y['test'], thr), mc.cls_metrics(Pv, y['val'], thr)
                mt.setdefault('f1_macro_thr', mt['f1_macro']); mt.setdefault('thr', float('nan'))
                res[f'{name}{"+TS" if ts else ""}'] = {'test': mt, 'val': mv, 'T': T if ts else None}
                lines.append(f'{name:38s} {"ya" if ts else "-":>5s} ' + ' '.join(f'{mt[k]:11.4f}' for k in keys)
                             + (f' {T:6.3f}' if ts else f' {"":6s}'))
        lines.append(f'(val AUC: ' + ', '.join(f'{k} {v["val"]["auc"]:.4f}' for k, v in res.items()
                                               if not k.endswith('+TS')) + ')')
        txt = '\n'.join(lines)
        print(txt)
        with open(out_txt, 'a') as f:
            f.write(txt + '\n\n')
        all_res[it] = res

    pd.DataFrame(kurva).to_csv(f'{pred_dir}/{a.varian}_evalbaca_kurva.csv', index=False)
    with open(f'{pred_dir}/{a.varian}_evalbaca.json', 'w') as f:
        json.dump({'args_eval': vars(cli), 'N': a.num_trials, 'hasil': all_res}, f, indent=2)
    print(f'Selesai. Ringkasan: {out_txt} | kurva sigma: {pred_dir}/{a.varian}_evalbaca_kurva.csv')


if __name__ == '__main__':
    main()