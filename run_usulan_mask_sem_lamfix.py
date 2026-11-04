"""
Ablasi bobot — masking (Mod 2) + Stochastic EM (Mod 3) + bobot TETAP pada y (lambda konstan).

Tujuan: menguji apakah "perhatian lebih ke y" masih berguna setelah masking. LBTW (adaptif) di atas
masking hanya memberi lambda_y ~ 1.03; di sini lambda dibuat jauh lebih kuat dan TETAP selama training.
    L = L_fitur + lambda * L_y      (lambda ditentukan sebelum training, tidak berubah)
Dasar: skalarisasi linear multi-task; Xin dkk. NeurIPS 2022 ("Do Current Multi-Task Optimization
Methods in Deep Learning Even Help?"); Kurin dkk. NeurIPS 2022 ("In Defense of the Unitary
Scalarization for Deep Multi-Task Learning").

Lambda hanya dikalikan ke sel y (di baris kondisional maupun full noise); semua fitur berbobot 1.
Early stopping & scheduler memakai loss TANPA bobot (rata-rata di sel c = 1), sama dengan varian lain.
Masking, E-step (SEM 1 sampel, mode asli), dan test sama persis dengan run_usulan_mask_sem.py.

Hasil: results/ablasi_mask_sem_lamfix/{ds}/{tag}/seed{k}/usulan_mask_sem_lam10_p0.5_*
       (ckpt/ablasi_mask_sem_lamfix/...)

Contoh:
  python run_usulan_mask_sem_lamfix.py --dataset bike --ratio 0.3 --seed 0 --quick
  python run_usulan_mask_sem_lamfix.py --dataset bike --ratio 0.3 --seed 0              # lambda = 10 (default ablasi)
  python run_usulan_mask_sem_lamfix.py --dataset bike --ratio 0.3 --seed 0 --lam 10
"""

import torch
from tqdm import tqdm

from run_usulan import build_parser
from run_usulan_mask import build_net, make_cond, edm_loss_mask, add_mask_args, run_mask, varian_suffix


def train_mstep_mask_lamfix(train_data, train_mask, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    """M-step masking dengan bobot tetap lambda pada kolom y. Kembali (net, None)."""
    X = torch.tensor(train_data, dtype=torch.float32, device=device)
    M = torch.tensor(train_mask, dtype=torch.float32, device=device)
    net = build_net(dim, args, device, print_model)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.9, patience=50)

    col_w = torch.ones(dim, device=device)
    col_w[y_idx] = args.lam

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
            err = edm_loss_mask(net, X[idx], c)                    # [B, d], 0 di sel bersih
            loss = (err * col_w).sum() / c.sum()                   # loss berbobot untuk update
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_err += err.detach().sum().item()
            tot_cnt += c.sum().item()
        cur = tot_err / tot_cnt                                    # loss TANPA bobot
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


def main():
    p = add_mask_args(build_parser('Ablasi: masking + SEM + bobot tetap lambda pada y'))
    p.add_argument('--lam', type=float, default=10.0, help='bobot tetap kolom y (1 = tanpa bobot)')
    p.set_defaults(num_trials=10)            # ablasi: N = 10 di semua baris (sama dengan metode usulan)
    args = p.parse_args()
    args.sem = True
    if args.varian == 'auto':
        args.varian = f'usulan_mask_sem_lam{args.lam:g}_{varian_suffix(args)}'
    run_mask(args, train_mstep_mask_lamfix, f'bobot tetap lambda_y = {args.lam:g}',
             group='ablasi_mask_sem_lamfix')


if __name__ == '__main__':
    main()