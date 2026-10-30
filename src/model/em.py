"""
Loop EM (DiffPuter + stochastic EM)

Tugas:
    - Inisialisasi fitur bolong di train dengan mean
    - Ulangi K kali: M-step (latih diffusion) lalu E-step
    - E-step: isi fitur bolong dengan 1 sampel acak (stochastic EM) atau rata-rata (DiffPuter asli)
    - y di train selalu diketahui (penanda 0 saat E-step)

Input:
    data train (+ val untuk model final), configs/experiment.yaml, nama varian

Output:
    Model final di results/checkpoints/, log waktu di results/logs/

Catatan:
    Rujukan: diffputer_asli/main.py (baris ~188 untuk E-step).
"""


def main():
    raise NotImplementedError("Belum diimplementasikan")


if __name__ == "__main__":
    main()
