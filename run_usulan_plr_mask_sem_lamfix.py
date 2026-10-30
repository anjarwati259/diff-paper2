"""
Varian — embedding numerik PERIODIK (PLR) + masking (Mod 2) + Stochastic EM (Mod 3) + bobot tetap lambda pada y.

Masalah: denoiser MLP DiffPuter menerima tiap kolom numerik sebagai SATU skalar lewat satu lapisan
linear. Jaringan saraf punya "spectral bias" (Rahaman dkk., ICML 2019): mudah belajar fungsi halus,
sulit menangkap perubahan tajam yang umum di data tabular (Grinsztajn dkk., NeurIPS 2022), mis.
puncak sewa sepeda di jam 8 & 17 pada hari kerja.

Perbaikan: embedding periodik PLR (Gorishniy, Rubachev & Babenko, NeurIPS 2022,
"On Embeddings for Numerical Features in Tabular Deep Learning"), untuk tiap kolom numerik j
(fitur numerik dan y):
    v_j = [sin(2*pi*c_j1*x), cos(2*pi*c_j1*x), ..., sin(2*pi*c_jK*x), cos(2*pi*c_jK*x)]   (c_jk dipelajari,
          inisialisasi N(0, sigma^2))
    e_j = ReLU(W_j v_j + b_j)                                                              (per kolom)
Lalu [e_1, ..., e_m, bit kategorik, penanda c] -> Linear -> MLP DiffPuter (tidak berubah).
Bit kategorik tetap apa adanya (nilainya kontinu ber-noise, embedding tabel tidak bisa dipakai).

Yang lain SAMA dengan run_usulan_mask_sem_lamfix.py: masking p, loss EDM di sel c = 1, bobot tetap
lambda pada y (early stopping pakai loss tanpa bobot), E-step SEM 1 sampel (mode asli), test mode
kondisional. Precond: input embedding = c_in*x untuk sel ditebak, x apa adanya untuk sel bersih.

Hasil: results/plr_mask_sem_lamfix/{ds}/{tag}/seed{k}/usulan_plr_mask_sem_lam5_p0.5_*
       (ckpt/plr_mask_sem_lamfix/...)

Contoh:
  python run_usulan_plr_mask_sem_lamfix.py --dataset bike --ratio 0.3 --seed 0 --quick
  python run_usulan_plr_mask_sem_lamfix.py --dataset bike --ratio 0.3 --seed 0 --lam 5
  python run_usulan_plr_mask_sem_lamfix.py ... --plr_k 48 --plr_dim 24 --plr_sigma 1.0
"""

import json
import math

import torch
import torch.nn as nn
from tqdm import tqdm

from run_diffputer import PositionalEmbedding
from run_usulan import build_parser
from run_usulan_mask import PrecondMask, make_cond, edm_loss_mask, add_mask_args, run_mask, varian_suffix


# ======================================================================
# Embedding periodik PLR (Gorishniy dkk. 2022)
# ======================================================================
class PeriodicPLR(nn.Module):
    """x [B, m] -> [B, m * d_emb]. Parameter terpisah per kolom."""
    def __init__(self, n_cols, n_freq=48, d_emb=24, sigma=1.0):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(n_cols, n_freq) * sigma)          # c_jk
        bound = 1 / math.sqrt(2 * n_freq)
        self.W = nn.Parameter(torch.empty(n_cols, 2 * n_freq, d_emb).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(n_cols, d_emb).uniform_(-bound, bound))

    def forward(self, x):
        z = 2 * math.pi * self.freq[None] * x[..., None]                       # [B, m, K]
        v = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)                    # [B, m, 2K]
        e = torch.relu(torch.einsum('bmk,mke->bme', v, self.W) + self.b[None]) # [B, m, d_emb]
        return e.flatten(1)


class MLPDiffusionPLR(nn.Module):
    """MLPDiffusion DiffPuter + penanda c; kolom numerik masuk lewat PLR, bit kategorik apa adanya."""
    def __init__(self, d_in, num_idx, dim_t=512, n_freq=48, d_emb=24, sigma=1.0):
        super().__init__()
        self.num_idx = list(num_idx)
        self.other_idx = [j for j in range(d_in) if j not in set(self.num_idx)]
        self.plr = PeriodicPLR(len(self.num_idx), n_freq, d_emb, sigma)
        self.proj = nn.Linear(len(self.num_idx) * d_emb + len(self.other_idx) + d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, c, noise_labels):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        h = torch.cat([self.plr(x[:, self.num_idx]), x[:, self.other_idx], c], dim=1)
        return self.mlp(self.proj(h) + emb)


def build_net_plr(dim, y_idx, args, device, print_model):
    num_idx = list(range(args.n_num)) + [y_idx]                  # fitur numerik + y
    denoise_fn = MLPDiffusionPLR(dim, num_idx, args.hid_dim, args.plr_k, args.plr_dim,
                                 args.plr_sigma).to(device)
    if print_model:
        print(denoise_fn)
        print(f'PLR untuk {len(num_idx)} kolom numerik (termasuk y); '
              f'{len(denoise_fn.other_idx)} kolom bit kategorik apa adanya; '
              f'parameter = {sum(p.numel() for p in denoise_fn.parameters()):,}')
    return PrecondMask(denoise_fn).to(device)


# ======================================================================
# M-step: masking + bobot tetap lambda (sama dengan run_usulan_mask_sem_lamfix.py), jaringan PLR
# ======================================================================
def train_mstep_plr(train_data, train_mask, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    X = torch.tensor(train_data, dtype=torch.float32, device=device)
    M = torch.tensor(train_mask, dtype=torch.float32, device=device)
    net = build_net_plr(dim, y_idx, args, device, print_model)
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


def main():
    p = add_mask_args(build_parser('PLR + masking + SEM + bobot tetap lambda'))
    p.add_argument('--lam', type=float, default=5.0, help='bobot tetap kolom y')
    p.add_argument('--plr_k', type=int, default=48, help='jumlah frekuensi per kolom')
    p.add_argument('--plr_dim', type=int, default=24, help='dimensi embedding per kolom')
    p.add_argument('--plr_sigma', type=float, default=1.0, help='skala inisialisasi frekuensi')
    args = p.parse_args()
    args.sem = True
    with open(f'datasets/Info/{args.dataset}.json') as f:
        args.n_num = len(json.load(f)['num_cols'])
    if args.varian == 'auto':
        args.varian = f'usulan_plr_mask_sem_lam{args.lam:g}_{varian_suffix(args)}'
    info = (f'PLR (K={args.plr_k}, d={args.plr_dim}, sigma={args.plr_sigma:g}) + '
            f'bobot tetap lambda_y = {args.lam:g}')
    run_mask(args, train_mstep_plr, info, group='plr_mask_sem_lamfix')


if __name__ == '__main__':
    main()