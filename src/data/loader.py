"""
Pemuat data untuk eksperimen

Tugas:
    - Gabungkan fitur + y (y kolom terakhir), encode kategorik (one-hot / binary)
    - Standardisasi pakai statistik train (mean/std dari nilai teramati), lalu bagi 2 seperti kode asli
    - Kembalikan matriks, mask fitur, indeks kolom y, dan fungsi inverse ke skala asli

Input:
    data/processed/, data/splits/, data/masks/

Output:
    Objek data train/val/test siap pakai untuk src/model dan src/baselines

Catatan:
    Pengganti load_dataset di diffputer_asli/dataset.py (yang tidak memasukkan kolom target).
"""


def main():
    raise NotImplementedError("Belum diimplementasikan")


if __name__ == "__main__":
    main()
