"""
Rekap hasil SEMUA metode dengan ATURAN SAMA, langsung ke file EXCEL (.xlsx). Tidak perlu run ulang:
semua angka dibaca dari *_metrics.json yang sudah ada.  (20 Nov 2026)

KLASIFIKASI : Acc, F1 macro, AUC, Brier dari peluang MENTAH, kelas = ARGMAX (ambang 0.5) = sama dengan usulan.
              baseline (run_baseline.py / run_svm.py / run_xgboost_nan.py): test -> y_mentah (BUKAN f1_macro_thr / y_ts)
REGRESI     : RMSE, MAE, R2, CRPS (skala standar-log).   baseline: test -> rmse, mae, r2, crps
USULAN      : hasil -> pm_terakhir -> test (ganti dengan --baca, mis. akhir_terakhir / sampel_terakhir)
Jenis tugas dikenali otomatis dari isi file.

Folder dipindai relatif terhadap FOLDER PROYEK (folder yang berisi data_loader.py: folder skrip ini atau induknya),
jadi skrip boleh dijalankan dari folder proyek maupun dari baselines_eksternal/.
Bawaan --akar: results_zlog  baselines_eksternal/results_zlog  (tambahkan results_std dll. bila perlu).

KELUARAN (otomatis): {folder proyek}/rekap/rekap_{dataset}_{MECH}{rr}.xlsx  (atau --out nama.xlsx), berisi sheet:
  ringkasan  : rata-rata per metode (+ sd metrik utama, jumlah seed), diurutkan (AUC tertinggi / RMSE terkecil)
  per_seed   : semua nilai per metode per seed (satu baris per metode-seed)
  acc / f1_macro / auc / brier  (klasifikasi)  atau  rmse / mae / r2 / crps  (regresi):
               tabel metode x seed0..seed4 + rata-rata
Butuh paket openpyxl (pip install openpyxl).

Pemakaian:
  python rekap_f1_argmax.py --dataset shoppers --ratio 0.5
  python rekap_f1_argmax.py --dataset shoppers --ratio 0.7 --per_seed        # + tabel per seed di layar
  python rekap_f1_argmax.py --dataset bike --ratio 0.5 --out rekap_bike_MAR50.xlsx
"""
import os
import json
import glob
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = next((d for d in (_HERE, os.path.dirname(_HERE)) if os.path.isfile(os.path.join(d, 'data_loader.py'))),
                   os.getcwd())

KOLOM = {'cls': ['acc', 'f1_macro', 'auc', 'brier'], 'reg': ['rmse', 'mae', 'r2', 'crps']}
UTAMA = {'cls': ('auc', False), 'reg': ('rmse', True)}           # (metrik pengurut, naik?)


def baca(path, baca_usulan):
    """-> (nama, dict metrik test) atau (None, None)."""
    try:
        js = json.load(open(path))
    except Exception:
        return None, None
    if 'hasil' in js:                                                   # metode usulan
        h = js['hasil'].get(baca_usulan)
        if not isinstance(h, dict) or 'test' not in h:
            return None, None
        return js.get('varian', os.path.basename(path)), h['test']
    te = js.get('test')
    nama = js.get('method') or os.path.basename(path).replace('_metrics.json', '')
    if isinstance(te, dict) and 'y_mentah' in te:                       # baseline klasifikasi
        return nama, te['y_mentah']
    if isinstance(te, dict) and 'rmse' in te:                           # baseline regresi
        return nama, te
    return None, None


def tulis_excel(path, lembar):
    """lembar: dict nama_sheet -> DataFrame. Angka 4 desimal, lebar kolom menyesuaikan, baris judul dibekukan."""
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        for nama, d in lembar.items():
            d.to_excel(w, sheet_name=nama, index=False)
            ws = w.sheets[nama]
            ws.freeze_panes = 'B2'
            for j, kol in enumerate(d.columns, 1):
                huruf = get_column_letter(j)
                if pd.api.types.is_float_dtype(d[kol]):
                    for sel in ws[huruf][1:]:
                        sel.number_format = '0.0000'
                lebar = max([len(str(kol))] + [len(str(v)) for v in d[kol].astype(str).head(200)])
                ws.column_dimensions[huruf].width = min(max(lebar, 8) + 2, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ratio', type=float, default=0.3)
    ap.add_argument('--mech', default='MAR')
    ap.add_argument('--akar', nargs='+', default=['results_zlog', 'baselines_eksternal/results_zlog'],
                    help='folder induk hasil (relatif terhadap folder proyek), dipindai rekursif')
    ap.add_argument('--baca', default='pm_terakhir', help='cara baca usulan (pm_terakhir, akhir_terakhir, ...)')
    ap.add_argument('--out', default=None,
                    help='nama file Excel (bawaan: rekap/rekap_{dataset}_{MECH}{rr}.xlsx di folder proyek)')
    ap.add_argument('--per_seed', action='store_true', help='cetak juga tabel per seed di layar')
    a = ap.parse_args()

    os.chdir(PROJECT_DIR)
    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    rows = []
    for akar in a.akar:
        if not os.path.isdir(akar):
            print(f'[lewati] folder {os.path.abspath(akar)} tidak ada')
            continue
        for f in glob.glob(f'{akar}/**/{a.dataset}/{tag}/seed*/*_metrics.json', recursive=True):
            nama, m = baca(f, a.baca)
            if m is None:
                continue
            task = 'cls' if 'auc' in m else ('reg' if 'rmse' in m else None)
            if task is None:
                continue
            seed = int(os.path.basename(os.path.dirname(f)).replace('seed', ''))
            grup = os.path.relpath(f, akar).split(os.sep)[0]
            rows.append({'metode': f'{grup}/{nama}', 'seed': seed, 'task': task,
                         **{k: float(m.get(k, np.nan)) for k in KOLOM[task]}, 'file': f})
    if not rows:
        print(f'Tidak ada hasil untuk {a.dataset} {tag} di {[os.path.abspath(x) for x in a.akar]}')
        return
    df = pd.DataFrame(rows).drop_duplicates(['metode', 'seed'], keep='last').sort_values(['metode', 'seed'])
    task = df['task'].mode()[0]
    df = df[df.task == task]
    kol = KOLOM[task]
    urut, naik = UTAMA[task]
    pd.set_option('display.width', 220, 'display.max_rows', 500)
    aturan = 'peluang mentah, kelas = argmax' if task == 'cls' else 'skala standar-log'
    print(f'=== {a.dataset} {tag} ({"klasifikasi" if task == "cls" else "regresi"}) | aturan: {aturan} | '
          f'usulan: {a.baca} ===\n')
    agg = {'n_seed': ('seed', 'count'), 'seed': ('seed', lambda s: ','.join(map(str, s)))}
    agg.update({k: (k, 'mean') for k in kol})
    agg[f'{urut}_sd'] = (urut, lambda x: x.std(ddof=0))
    ring = df.groupby('metode').agg(**agg).sort_values(urut, ascending=naik)
    print(ring.round(4).to_string())

    pivot = {}
    for k in kol:
        t = df.pivot(index='metode', columns='seed', values=k)
        t.columns = [f'seed{c}' for c in t.columns]
        t['rata'] = t.mean(axis=1)
        pivot[k] = t.loc[ring.index]
        if a.per_seed:
            print(f'\n--- {k} per seed ---')
            print(pivot[k].round(4).to_string())
    kurang = ring[ring.n_seed < 5]
    if len(kurang):
        print(f'\nPERHATIAN: belum 5 seed -> {", ".join(kurang.index)}')

    out = a.out or os.path.join('rekap', f'rekap_{a.dataset}_{tag}.xlsx')
    if not out.lower().endswith('.xlsx'):
        out = os.path.splitext(out)[0] + '.xlsx'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    lembar = {'ringkasan': ring.reset_index(),
              'per_seed': df.drop(columns=['file', 'task']).reset_index(drop=True)}
    lembar.update({k: v.reset_index() for k, v in pivot.items()})
    try:
        tulis_excel(out, lembar)
    except ImportError:
        print('\nPaket openpyxl belum terpasang -> pip install openpyxl, lalu jalankan lagi.')
        return
    print(f'\nFile Excel disimpan: {os.path.abspath(out)}')


if __name__ == '__main__':
    main()