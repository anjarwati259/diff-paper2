"""
Persiapan dataset

Tugas:
    - Baca data/raw/{dataset}, buang kolom sesuai configs/datasets.yaml
    - Rapikan kategorik, log-transform target bila perlu
    - News: buang url & timedelta, ambil sampel 20 ribu baris (seed 1234)
    - Jangan buang baris ber-NaN asli (studi kasus)

Input:
    data/raw/{dataset}/, configs/datasets.yaml

Output:
    data/processed/{dataset}/data.csv, info.json

Catatan:
    Bedanya dengan download_and_process.py asli: tidak membuang baris ber-NaN dan target ikut disimpan.
"""


def main():
    raise NotImplementedError("Belum diimplementasikan")


if __name__ == "__main__":
    main()
