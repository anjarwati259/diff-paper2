"""
Ablasi #5 — Modifikasi 2 (target-aware masking) + Modifikasi 3 (Stochastic EM), TANPA bobot.

Masking, training, dan sampling test sama persis dengan run_usulan_mask.py.
Bedanya hanya E-step di data train: SATU sampel acak (1 trial, mode full noise seperti asli),
bukan rata-rata num_trials sampel.

Hasil: results/mod2_masking_sem/{ds}/{tag}/seed{k}/usulan_mask_sem_p0.5_*   (ckpt/mod2_masking_sem/...)

Contoh:
  python run_usulan_mask_sem.py --dataset bike --ratio 0.3 --seed 0 --quick
  python run_usulan_mask_sem.py --dataset bike --ratio 0.3 --seed 0
"""

from run_usulan import build_parser
from run_usulan_mask import train_mstep_mask, add_mask_args, run_mask, varian_suffix


def main():
    p = add_mask_args(build_parser('Ablasi: masking (Mod 2) + Stochastic EM (Mod 3)'))
    p.set_defaults(num_trials=10)            # ablasi: N = 10 di semua baris (sama dengan metode usulan)
    args = p.parse_args()
    args.sem = True
    if args.varian == 'auto':
        args.varian = f'usulan_mask_sem_{varian_suffix(args)}'
    run_mask(args, train_mstep_mask, 'tanpa bobot', group='mod2_masking_sem')


if __name__ == '__main__':
    main()