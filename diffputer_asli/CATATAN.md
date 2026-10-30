Salinan kode resmi DiffPuter (Zhang dkk., ICLR 2025). JANGAN diubah — dipakai sebagai rujukan
dan untuk membandingkan dengan versi modifikasi di `src/model/`.

Temuan penting dari kode ini (lihat rancangan):
- `dataset.py`: kolom target (target_col_idx) TIDAK dimasukkan ke matriks diffusion.
- `diffusion_utils.py` EDMLoss: noise dan loss di semua kolom dengan bobot sama.
- `diffusion_utils.py` impute_mask: sel teramati = asli + noise level t_next; resampling 20x per langkah.
- `main.py` baris ~188: E-step memakai rata-rata sampel (diganti stochastic EM di versi usulan).
- `generate_mask.py` MAR_mask: dipanggil terpisah untuk train dan test (diganti: mekanisme sekali dari train).
- `download_and_process.py`: split 70/30 seed 1234 tanpa validasi; baris ber-NaN asli dibuang.
- Kategorik di kode memakai binary/analog bits (paper menulis one-hot).
