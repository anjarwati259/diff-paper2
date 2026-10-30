"""
Pembuatan mask missing (hanya fitur)

Tugas:
    - MAR 0–40% (utama) dan MCAR 30% (pembanding)
    - MAR: tentukan mekanisme SEKALI dari train (kolom selalu teramati, bobot, intercept),
      lalu terapkan ke train, val, test
    - Hitung dan simpan proporsi missing aktual

Input:
    data/processed/, data/splits/, configs/experiment.yaml

Output:
    data/masks/{dataset}/{MAR|MCAR}{ratio}/seed{k}/{train,val,test}_mask.npy + mekanisme.json

Catatan:
    Dasar fungsi: MAR_mask di diffputer_asli/generate_mask.py, dipecah jadi buat_mekanisme() dan terapkan_mekanisme().
"""


def main():
    raise NotImplementedError("Belum diimplementasikan")


if __name__ == "__main__":
    main()
