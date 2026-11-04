"""
EVALUASI ULANG checkpoint metode usulan (REGRESI) dengan prediksi titik POSTERIOR MEAN — TANPA training ulang.

Model tiap iterasi EM dimuat dari ckpt/, lalu sampling kondisional (val & test) dijalankan lagi sambil MEREKAM
output denoiser D(x_t; sigma) kolom y di setiap langkah sampling (sampler perekam yang sama dengan jalur
klasifikasi run_usulan_final.py).

Cara membaca prediksi y yang dibandingkan (semua di SKALA STANDAR-LOG, sama dengan eval_zlog.py):
  sampel       : rata-rata N sampel akhir (= cara run_usulan_final.py sekarang); CRPS & coverage dari sampel ini
  pm           : POSTERIOR MEAN. EDM: D(x_t; sigma) = E[y | x_t]; rata-rata atas N lintasan -> E[y | x_obs].
                 Jauh lebih halus daripada rata-rata sampel akhir (noise sampling berkurang). Langkah sigma
                 DIPILIH DI VAL (RMSE val terkecil, --select); kurva semua langkah disimpan.
  ensemble     : gabungan semua iterasi EM yang dievaluasi
                   sampel  -> semua sampel digabung (sama dengan baris [ensemble iterasi] eval_zlog.py)
                   pm      -> kurva pm dirata-rata antar iterasi, langkah sigma dipilih di val
Prediksi titik pm hanya mengganti PREDIKSI TITIK (RMSE/MAE/R2); interval & CRPS tetap dari sampel.

Pemakaian (argumen menunjuk run yang sudah selesai; setting dibaca dari {varian}_metrics.json):
  python eval_ckpt_reg.py --dataset bike --ratio 0.3 --seed 0
  python eval_ckpt_reg.py --dataset bike --ratio 0.3 --seed 0 --iters 3 4 --num_trials 10
Hasil (folder run yang sama, tidak menimpa hasil run):
  results_zlog/usulan_final/{ds}/{tag}/seed{k}/{varian}_evalpm.txt, _evalpm.json, _evalpm_kurva.csv,
  _evalpm_{val,test}_pm.npy (prediksi titik pm terpilih, skala standar-log; iterasi tunggal terakhir & ensemble)
"""

import os
import json
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from data_loader import load_split, clip_bounds, inverse_y, y_true_z
from run_diffputer import y_metrics
from run_usulan_final import build_net, GROUP, run_trials_cond_record

PARTS = ('val', 'test')


def point(y, p):
    return {'rmse': float(np.sqrt(((y - p) ** 2).mean())), 'mae': float(np.abs(y - p).mean()),
            'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def pick(curve_val, y_val, how):
    """curve_val [steps, n] -> indeks langkah terbaik di val."""
    sc = [point(y_val, c)[how] for c in curve_val]
    return int(np.nanargmax(sc)) if how == 'r2' else int(np.nanargmin(sc))


def main():
    p = argparse.ArgumentParser(description='Evaluasi ulang checkpoint regresi: posterior mean (tanpa training ulang)')
    p.add_argument('--dataset', required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mech', default='MAR')
    p.add_argument('--varian', default='usulan_plr_mask_sem_lam10_p0.5')
    p.add_argument('--out_dir', default='results_zlog')
    p.add_argument('--iters', type=int, nargs='*', default=None, help='iterasi EM (default: semua yang ada)')
    p.add_argument('--num_trials', type=int, default=None, help='N lintasan (default: sama dengan run)')
    p.add_argument('--chunk_rows', type=int, default=None)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--eval_seed', type=int, default=123, help='seed sampling evaluasi ulang')
    p.add_argument('--select', default='rmse', choices=['rmse', 'mae'], help='kriteria memilih langkah sigma di val')
    cli = p.parse_args()

    tag = f'{cli.mech}{int(round(cli.ratio * 100)):02d}'
    pred_dir = f'{cli.out_dir}/{GROUP}/{cli.dataset}/{tag}/seed{cli.seed}'
    with open(f'{pred_dir}/{cli.varian}_metrics.json') as f:
        run_summary = json.load(f)
    a = SimpleNamespace(**run_summary['args'])
    if getattr(a, 'task', 'regression') != 'regression':
        raise SystemExit('skrip ini untuk REGRESI (klasifikasi: eval_ckpt_cls.py)')
    n_run = run_summary['args']['num_trials']
    if cli.num_trials:
        a.num_trials = cli.num_trials
    if cli.chunk_rows:
        a.chunk_rows = cli.chunk_rows
    if getattr(a, 'resample_test', None) is None:
        a.resample_test = a.resample
    ckpt_dir = f'ckpt/{GROUP}/{cli.dataset}/{tag}/seed{cli.seed}/{a.varian}/{n_run}_{a.num_steps}'
    iters = cli.iters if cli.iters else sorted(int(x) for x in os.listdir(ckpt_dir)
                                               if x.isdigit() and os.path.exists(f'{ckpt_dir}/{x}/model.pt'))
    device = f'cuda:{cli.gpu}' if torch.cuda.is_available() and cli.gpu >= 0 else 'cpu'

    data, meta = load_split(cli.dataset, cli.seed, cli.ratio, cli.mech, num_transform=getattr(a, 'num_transform', 'none'))
    yc = meta['y_cols']
    assert list(yc) == list(a.y_cols), 'struktur kolom y berbeda dengan run'
    y = {pt: y_true_z(data[pt]['df'], meta) for pt in PARTS}
    print(f'{cli.dataset} {tag} seed{cli.seed} | varian {a.varian} | iterasi {iters} | N={a.num_trials}, '
          f'langkah={a.num_steps}, resample={a.resample_test} | device {device}')

    out_txt = f'{pred_dir}/{a.varian}_evalpm.txt'
    open(out_txt, 'w').close()
    kurva, hasil = [], {}
    PMc = {pt: [] for pt in PARTS}           # kurva pm per iterasi [steps, n] (skala standar-log)
    SAM = {pt: [] for pt in PARTS}           # sampel akhir per iterasi [N, n]
    sig = None

    def report(label, pm_curve, samples):
        s = pick(pm_curve['val'], y['val'], cli.select)
        res = {}
        for pt in PARTS:
            ms = y_metrics(y[pt], samples[pt])
            res[pt] = {'sampel': ms, 'pm': {**point(y[pt], pm_curve[pt][s]), 'langkah': s, 'sigma': float(sig[s])}}
        lines = [f'=== {label} | langkah pm terpilih (val {cli.select}) = {s}, sigma {sig[s]:.3g} ===',
                 f'{"bagian":6s} {"cara baca":10s} {"RMSE":>8s} {"MAE":>8s} {"R2":>8s} {"CRPS":>8s} {"covmm":>7s}']
        for pt in PARTS:
            ms, mp = res[pt]['sampel'], res[pt]['pm']
            lines.append(f'{pt:6s} {"sampel":10s} {ms["rmse"]:8.4f} {ms["mae"]:8.4f} {ms["r2"]:8.4f} '
                         f'{ms["crps"]:8.4f} {ms["coverage_minmax"]:7.3f}')
            lines.append(f'{pt:6s} {"pm":10s} {mp["rmse"]:8.4f} {mp["mae"]:8.4f} {mp["r2"]:8.4f} '
                         f'{"(sampel)":>8s} {"":7s}')
        txt = '\n'.join(lines)
        print(txt)
        with open(out_txt, 'a') as f:
            f.write(txt + '\n\n')
        return res, s

    for it in iters:
        torch.manual_seed(cli.eval_seed + it); np.random.seed(cli.eval_seed + it)
        net = build_net(meta['dim'], a, device, False)
        net.load_state_dict(torch.load(f'{ckpt_dir}/{it}/model.pt', map_location=device))
        net.eval()
        cur_pm, cur_s = {}, {}
        for pt in PARTS:
            S, D, sig = run_trials_cond_record(net, data[pt]['X_obs'], data[pt]['mask'], yc, a, device,
                                               f'iter {it} {pt}', clip_bounds(meta))
            cur_s[pt] = inverse_y(S[:, :, yc[0]], meta)                   # [N, n] skala standar-log
            cur_pm[pt] = inverse_y(D[..., 0].mean(0), meta)               # [steps, n]
            PMc[pt].append(cur_pm[pt]); SAM[pt].append(cur_s[pt])
        for st in range(cur_pm['val'].shape[0]):
            row = {'iterasi': it, 'langkah': st, 'sigma': float(sig[st])}
            for pt in PARTS:
                m = point(y[pt], cur_pm[pt][st])
                row.update({f'{pt}_rmse': m['rmse'], f'{pt}_mae': m['mae'], f'{pt}_r2': m['r2']})
            kurva.append(row)
        hasil[f'iterasi {it}'], s_it = report(f'iterasi {it}', cur_pm, cur_s)
        if it == iters[-1]:
            for pt in PARTS:
                np.save(f'{pred_dir}/{a.varian}_evalpm_{pt}_pm_iter{it}.npy', cur_pm[pt][s_it].astype(np.float32))

    if len(iters) > 1:
        ens_pm = {pt: np.mean(PMc[pt], 0) for pt in PARTS}
        ens_s = {pt: np.concatenate(SAM[pt], 0) for pt in PARTS}
        hasil['ensemble iterasi'], s_e = report(f'ensemble iterasi {iters}', ens_pm, ens_s)
        for pt in PARTS:
            np.save(f'{pred_dir}/{a.varian}_evalpm_{pt}_pm_ensemble.npy', ens_pm[pt][s_e].astype(np.float32))

    pd.DataFrame(kurva).to_csv(f'{pred_dir}/{a.varian}_evalpm_kurva.csv', index=False)
    with open(f'{pred_dir}/{a.varian}_evalpm.json', 'w') as f:
        json.dump({'args_eval': vars(cli), 'N': a.num_trials, 'iterasi': iters, 'hasil': hasil}, f, indent=2)
    print(f'Selesai. Ringkasan: {out_txt} | kurva sigma: {pred_dir}/{a.varian}_evalpm_kurva.csv')


if __name__ == '__main__':
    main()