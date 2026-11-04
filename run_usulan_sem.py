"""
Ablasi #4 — Modifikasi 3 SAJA: Stochastic EM (SEM) pada DiffPuter asli.

Satu-satunya perbedaan dari DiffPuter asli (run_diffputer.py):
  E-step di data train mengisi sel yang hilang dengan SATU sampel acak (1 trial),
  bukan rata-rata num_trials sampel. Tujuannya menjaga variasi bersyarat agar model
  di iterasi berikutnya tidak terlalu yakin (interval prediksi tidak terlalu sempit).
  Rujukan: Celeux & Diebolt 1985 (Stochastic EM); Wei & Tanner 1990 (Monte Carlo EM).
Training (M-step), sampling, dan test sama dengan asli; di test y-hat tetap rata-rata
num_trials sampel dan interval dari persentil sampel.

Catatan: metrik imputasi IN-SAMPLE (train) akan terlihat lebih buruk karena memakai satu
sampel, bukan rata-rata. Yang dinilai adalah imputasi & prediksi di TEST.

Hasil: results/mod3_sem/{ds}/{tag}/seed{k}/usulan_sem_*   (ckpt/mod3_sem/...)

Contoh:
  python run_usulan_sem.py --dataset bike --ratio 0.3 --seed 0 --quick
  python run_usulan_sem.py --dataset bike --ratio 0.3 --seed 0
"""

from run_diffputer import train_mstep
from run_usulan import build_parser, run


def train_mstep_asli(train_data, dim, y_idx, args, device, ckpt_iter_dir, print_model):
    """M-step persis DiffPuter asli (tanpa bobot). Kembali (net, None)."""
    return train_mstep(train_data, dim, args, device, ckpt_iter_dir, print_model), None


def main():
    p = build_parser('Ablasi: DiffPuter + Stochastic EM (Mod 3 saja)')
    p.set_defaults(num_trials=10)            # ablasi: N = 10 di semua baris (sama dengan metode usulan)
    args = p.parse_args()
    args.sem = True
    if args.varian == 'auto':
        args.varian = 'usulan_sem'
    run(args, train_mstep_asli, 'tanpa bobot, tanpa masking', group='mod3_sem')


if __name__ == '__main__':
    main()