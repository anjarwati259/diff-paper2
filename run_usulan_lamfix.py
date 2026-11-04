"""
Ablasi PEMBOBOTAN — DiffPuter asli + bobot TETAP pada y (lambda konstan), TANPA masking, TANPA SEM.

Baris tabel ablasi: "DIFFPUTER + BOBOT TETAP (10)". Pembanding langsung untuk DWA (run_usulan.py) dan
LBTW (run_usulan_lbtw.py): model, M-step, E-step (rata-rata trial, cara asli), sampling, dan metrik
SAMA PERSIS dengan run_usulan.py; satu-satunya beda adalah bobot kolom y.
    L = rata-rata( err_fitur * 1 , err_y * lambda )      (lambda tetap selama training)
Cara menghitung loss berbobot sama dengan DWA (rata-rata err * bobot kolom), sehingga yang dibandingkan
hanya ATURAN bobotnya (adaptif DWA / adaptif LBTW / tetap).
Dasar: skalarisasi linear multi-task (Xin dkk. & Kurin dkk., NeurIPS 2022).
Early stopping & LR scheduler memakai loss TANPA bobot (sama dengan asli dan varian lain).

Evaluasi = run_usulan.run(): skala standar-log, sampel y per iterasi disimpan (ensemble antar iterasi
dihitung di eval_zlog.py), hasil di results_zlog/mod1_lamfix/{ds}/{tag}/seed{k}/usulan_lamfix_lam10_*.
Default N = 10 sampel (sama dengan semua baris ablasi & metode usulan).

Contoh:
  python run_usulan_lamfix.py --dataset bike --ratio 0.3 --seed 0 --parts val test
  python run_usulan_lamfix.py --dataset bike --ratio 0.3 --seed 0 --parts val test --quick   # uji alur
"""
import numpy as np
import torch
from tqdm import tqdm

from run_diffputer import MLPDiffusion, Precond
from run_usulan import edm_loss_per_col, build_parser, run


def train_mstep_lamfix(train_data, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    """= train_mstep_dwa (run_usulan.py) dengan lambda_y TETAP. Kembali (net, riwayat lambda per epoch)."""
    torch_data = torch.tensor(train_data, dtype=torch.float32, device=device)
    denoise_fn = MLPDiffusion(dim, args.hid_dim).to(device)
    if print_model:
        print(denoise_fn)
    net = Precond(denoise_fn).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)

    col_w = torch.ones(dim, device=device)
    col_w[y_idx] = args.lam
    lam_hist = []

    best, patience = float('inf'), 0
    n = torch_data.shape[0]
    net.train()
    pbar = tqdm(range(args.max_epochs), desc='Training')
    for epoch in pbar:
        lam_hist.append(args.lam)
        perm = torch.randperm(n, device=device)
        tot_unw = 0.0
        for s in range(0, n, args.batch_size):
            batch = torch_data[perm[s:s + args.batch_size]]
            err = edm_loss_per_col(net, batch)              # [B, d]
            loss = (err * col_w).mean()                     # loss berbobot untuk update
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_unw += err.detach().mean().item() * batch.shape[0]
        cur = tot_unw / n                                   # loss TANPA bobot (seperti asli)
        sched.step(cur)
        if cur < best:
            best, patience = cur, 0
            torch.save(net.state_dict(), f'{ckpt_iter_dir}/model.pt')
        else:
            patience += 1
            if patience == args.patience:
                print('Early stopping')
                break
        pbar.set_postfix(loss=cur, lam_y=args.lam)
    pbar.close()
    net.load_state_dict(torch.load(f'{ckpt_iter_dir}/model.pt', map_location=device))
    net.eval()
    return net, np.array(lam_hist)


def main():
    p = build_parser('Ablasi pembobotan: DiffPuter + bobot tetap lambda pada y (tanpa masking, tanpa SEM)')
    p.add_argument('--lam', type=float, default=10.0, help='bobot tetap kolom y')
    p.set_defaults(num_trials=10)                          # N sama dengan semua baris ablasi & usulan
    args = p.parse_args()
    if args.varian == 'auto':
        args.varian = f'usulan_lamfix_lam{args.lam:g}'
    run(args, train_mstep_lamfix, f'bobot tetap lambda_y = {args.lam:g}, tanpa masking, tanpa SEM',
        group='mod1_lamfix')


if __name__ == '__main__':
    main()
