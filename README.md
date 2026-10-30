# Paper 2 — DiffPuter untuk Imputasi Fitur sekaligus Prediksi Target Regresi

Kerangka proyek untuk menjalankan rancangan yang sudah dikunci:
DiffPuter (ICLR 2025) diadaptasi agar mengisi fitur yang hilang **dan** memprediksi
target regresi y dalam satu model, dengan tiga modifikasi:

1. **Bobot loss pada y** (λ) — `src/model/diffusion_utils.py` (EDMLoss)
2. **Target-aware masking** (p, penanda m) — `src/model/diffusion_utils.py`, `src/model/model.py`
3. **Stochastic EM** (E-step training pakai 1 sampel acak) — `src/model/em.py`

Status: **kerangka saja**. File di `src/` berisi keterangan tugas (docstring) dan
belum diimplementasikan. Kode asli DiffPuter ada di `diffputer_asli/` sebagai rujukan.

---

## Struktur folder

```
paper2-diffputer-regresi/
├── README.md                  ← file ini
├── requirements.txt           ← paket Python
├── .gitignore
├── configs/
│   ├── datasets.yaml          ← 5 dataset + studi kasus, kolom target, kolom dibuang, log target
│   └── experiment.yaml        ← split, seed, missing, hyperparameter, varian ablation, metrik
├── data/
│   ├── raw/                   ← file unduhan asli (tidak diubah)
│   ├── processed/             ← data bersih per dataset: data.csv + info.json
│   ├── splits/                ← indeks train/val/test per seed
│   └── masks/                 ← mask missing per dataset / mekanisme / ratio / seed
├── diffputer_asli/            ← salinan kode resmi DiffPuter (JANGAN diubah, untuk rujukan)
├── baselines_eksternal/
│   ├── GRAPE/                 ← dari repo DiffPuter (ada train_y.py untuk prediksi label)
│   └── hyperimpute/           ← dari repo DiffPuter (mean, KNN, MICE, MissForest, dll.)
├── src/
│   ├── data/                  ← persiapan dataset, split, mask
│   ├── model/                 ← DiffPuter yang dimodifikasi
│   ├── baselines/             ← runner semua baseline (format output seragam)
│   └── eval/                  ← metrik, rekap tabel, uji statistik, grafik
├── scripts/                   ← perintah siap jalan per tahap (pilot, utama, ablation, dst.)
├── results/
│   ├── predictions/           ← CSV hasil prediksi + .npy sampel (sumber semua tabel)
│   ├── checkpoints/           ← bobot model (besar, tidak di-commit)
│   ├── logs/                  ← log training / sampling / waktu
│   ├── tables/                ← tabel hasil untuk naskah
│   └── figures/               ← grafik untuk naskah
├── notebooks/                 ← eksplorasi data dan cek cepat
└── paper/                     ← naskah, referensi, PPT rancangan
```

---

## Urutan kerja

| Tahap | Yang dikerjakan | File / folder |
|---|---|---|
| 0. Lingkungan | Buat environment, install paket, cek GPU | `requirements.txt` |
| 1. Data | Unduh 5 dataset (+ studi kasus) ke `data/raw/`, lalu bersihkan | `src/data/prepare_datasets.py` → `data/processed/` |
| 2. Split | Pecah 70/10/20 untuk seed 0–4 | `src/data/make_splits.py` → `data/splits/` |
| 3. Mask | MAR 0–40% + MCAR 30%, hanya fitur; mekanisme MAR dibuat sekali dari train | `src/data/make_masks.py` → `data/masks/` |
| 4. Adaptasi kode | Kolom target masuk matriks, bobot λ, masking p, stochastic EM, sampling batch | `src/model/` |
| 5. Pilot | Asli vs +masking vs +SEM vs lengkap; 1 dataset; MAR 10% & 40%; 3 seed | `scripts/run_pilot.sh` |
| 6. Tuning | Pilih λ dan p di data validasi (1–2 dataset), lalu dipakai tetap | `scripts/run_tuning.sh` |
| 7. Eksperimen inti | E1 (MAR 30%), E2 (MAR 0–40%), E4 (ablation) | `scripts/run_main.sh`, `scripts/run_ablation.sh` |
| 8. Baseline | Semua baseline dengan split dan mask yang SAMA | `scripts/run_baselines.sh` |
| 9. Pendukung | E3 MCAR 30%, E5 sensitivitas, E6 iterasi EM, E7 ketidakpastian, E8 imputasi, E9 waktu, E10 studi kasus | `scripts/` |
| 10. Rekap | Tabel, grafik, Friedman + Nemenyi (E11) | `src/eval/` → `results/tables/`, `results/figures/` |

Kalau hasil pilot belum menunjukkan peningkatan, opsi yang ditunda: residual target ala CARD
(lihat rancangan di project "PAPER 2").

---

## Konvensi penamaan

- Mask: `data/masks/{dataset}/{MAR|MCAR}{ratio}/seed{k}/{train|val|test}_mask.npy`
  (ratio ditulis dua digit: `MAR00`, `MAR10`, …, `MAR40`, `MCAR30`)
- Prediksi: `results/predictions/{dataset}/{MAR|MCAR}{ratio}/seed{k}/{metode}.csv`
- Sampel: file yang sama dengan akhiran `_samples.npy` (ukuran baris_test × N)
- Nama metode: `diffputer_asli`, `usulan_bobot`, `usulan_masking`, `usulan_sem`, `usulan_lengkap`,
  `mean_xgb`, `knn_xgb`, `mice_xgb`, `missforest_xgb`, `diffputer_xgb`, `xgb_nan`,
  `grape`, `neumiss`, `card`, `treeffuser`, `xgb_quantile`

## Format CSV prediksi (semua metode)

| id | {fitur asli, NaN jika hilang} | {fitur}_isian | y_asli | y_pred | y_low | y_high |
|---|---|---|---|---|---|---|

- `y_pred` = rata-rata N = 20 sampel (skala asli; inverse log per sampel dulu)
- `y_low`, `y_high` = persentil 5–95 (atau min–max sampel); kosongkan untuk metode tanpa interval
- `{fitur}_isian` opsional, hanya untuk evaluasi imputasi pendukung (E8)

## Aturan penting

- y di train **selalu lengkap**; y di val/test dikosongkan seluruhnya saat prediksi.
- Mask hanya untuk fitur, dibuat dari matriks fitur tanpa y.
- File mask dan split per seed dipakai **sama** oleh semua metode.
- Hyperparameter dipilih di val, model final dilatih ulang di train + val, dievaluasi di test.
