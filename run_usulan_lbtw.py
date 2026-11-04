"""
Metode USULAN — Modifikasi 1 versi LBTW satu arah (pembanding untuk DWA satu arah).

Dasar: Loss-Balanced Task Weighting, Liu, Liang & Gitter, AAAI 2019, Algorithm 1:
    untuk tiap batch B dan tugas t:  w_t = ( l_(B,t) / l_(0,i,t) )^alpha
    l_(0,i,t) = loss tugas t pada batch PERTAMA epoch i; loss tiap tugas dikali w_t (tanpa normalisasi).
    alpha = 0.5 terbaik di paper (juga dicoba 0.1).

Versi satu arah (dua kelompok: fitur dan y):
    rasio_k  = L_k(batch sekarang) / L_k(acuan)            (L = rata-rata loss PER KOLOM, tanpa gradien)
    lambda_y = max(1, (rasio_y / rasio_fitur)^alpha)        (= bobot LBTW y dibagi bobot LBTW fitur)
    bobot fitur = 1                                          (dikunci)
  -> perhatian hanya bisa bertambah ke y. Bobot diperbarui SETIAP BATCH (sesuai paper).

Pilihan acuan (--lbtw_ref):
    epoch : batch pertama tiap epoch (SESUAI PAPER). Catatan: dengan batch 4096, Bike hanya ~3 batch/epoch,
            sehingga rasio di dalam satu epoch dekat 1 dan efeknya bisa kecil.
    start : batch pertama M-step (awal training model di iterasi EM itu). Modifikasi dari paper;
            sinyal lebih kuat karena membandingkan dengan kondisi awal training.

Alur, model, sampling, metrik: sama persis dengan run_usulan.py / run_diffputer.py.
Early stopping & LR scheduler memakai loss TANPA bobot (sama dengan asli).

Nama hasil: usulan_lbtw_epoch_a0.5 atau usulan_lbtw_start_a0.5 (berbeda dari usulan_dwa dan diffputer_asli)
  results/mod1_lbtw/{ds}/{tag}/seed{k}/usulan_lbtw_{ref}_a{alpha}_*  (+ _lambda_iter{i}.npy: rata-rata lambda_y per epoch)

Contoh:
  python run_usulan_lbtw.py --dataset bike --ratio 0.3 --seed 0 --quick
  python run_usulan_lbtw.py --dataset bike --ratio 0.3 --seed 0 --parts val test   # acuan start (default ablasi)
  python run_usulan_lbtw.py --dataset bike --ratio 0.3 --seed 0 --lbtw_ref epoch   # acuan epoch (paper)
"""

import numpy as np
import torch
from tqdm import tqdm

from run_diffputer import MLPDiffusion, Precond
from run_usulan import edm_loss_per_col, build_parser, run


def train_mstep_lbtw(train_data, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    torch_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    denoise_fn = MLPDiffusion(dim, args.hid_dim).to(device)
    if print_model:
        print(denoise_fn)
    net = Precond(denoise_fn).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)

    feat_cols = [j for j in range(dim) if j != y_idx]
    ref = None                                   # (L_fitur_acuan, L_y_acuan)
    lam_hist = []

    best, patience = float('inf'), 0
    n = torch_data.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        if args.lbtw_ref == 'epoch':
            ref = None                           # acuan diambil ulang di batch pertama epoch ini
        perm = torch.randperm(n, device=device)
        tot_unw, lam_epoch = 0.0, []
        for s in range(0, n, args.batch_size):
            batch = torch_data[perm[s:s + args.batch_size]]
            err = edm_loss_per_col(net, batch)                  # [B, d]

            with torch.no_grad():
                per_col = err.mean(0)
                L_f = per_col[feat_cols].mean().item()
                L_y = per_col[y_idx].item()
            if ref is None:
                ref = (L_f, L_y)
            r_f = L_f / max(ref[0], 1e-12)
            r_y = L_y / max(ref[1], 1e-12)
            lam = max(1.0, (r_y / max(r_f, 1e-12)) ** args.lbtw_alpha)
            lam_epoch.append(lam)

            col_w = torch.ones(dim, device=device)
            col_w[y_idx] = lam
            loss = (err * col_w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_unw += err.detach().mean().item() * batch.shape[0]

        lam_hist.append(float(np.mean(lam_epoch)))
        cur = tot_unw / n                                       # loss TANPA bobot (seperti asli)
        sched.step(cur)
        if cur < best:
            best, patience = cur, 0
            torch.save(net.state_dict(), f'{ckpt_iter_dir}/model.pt')
        else:
            patience += 1
            if patience == args.patience:
                print('Early stopping')
                break
        pbar.set_postfix(loss=cur, lam_y=round(lam_hist[-1], 3))
    pbar.close()
    net.load_state_dict(torch.load(f'{ckpt_iter_dir}/model.pt', map_location=device))
    net.eval()
    return net, np.array(lam_hist)


def main():
    p = build_parser('Metode usulan: DiffPuter + LBTW satu arah')
    p.add_argument('--lbtw_alpha', type=float, default=0.5, help='alpha LBTW (paper: 0.5 terbaik)')
    p.add_argument('--lbtw_ref', choices=['epoch', 'start'], default='start',
                   help="acuan loss: 'epoch' = batch pertama tiap epoch (paper), 'start' = awal M-step")
    p.set_defaults(num_trials=10)            # ablasi: N = 10 di semua baris (sama dengan metode usulan)
    args = p.parse_args()
    if args.varian == 'auto':
        args.varian = f'usulan_lbtw_{args.lbtw_ref}_a{args.lbtw_alpha:g}'
    info = f'LBTW satu arah: alpha={args.lbtw_alpha:g}, acuan={args.lbtw_ref}'
    run(args, train_mstep_lbtw, info, group='mod1_lbtw')


if __name__ == '__main__':
    main()