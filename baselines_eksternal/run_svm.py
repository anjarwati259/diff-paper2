"""
BASELINE DUA TAHAP — TAHAP 2: PREDIKSI DENGAN SVM dari data yang sudah diimputasi (CSV run_imputasi.py).
Regresi -> SVR, klasifikasi -> SVC (tugas dibaca otomatis dari dataset).

ALUR (pemodelan klasik)
  1. Baca results/imputasi/{imputer}/{dataset}/MAR{rr}/seed{k}/imputed_{train,val,test}.csv (nilai asli),
     hasil run_imputasi.py (mean, knn, mice, missforest, hyperimpute, miwae) atau run_imputasi_diffputer.py (diffputer).
  2. Persiapan fitur, aturan sama dengan model_classification.py / run_xgboost_nan.py:
       kolom kategorik dideteksi dari dtype (teks/bool) -> one-hot drop_first (kolom val/test disamakan ke train),
       lalu StandardScaler yang di-fit di TRAIN.
  3. Model dengan setting bawaan scikit-learn (tanpa tuning), dilatih di TRAIN saja:
       regresi     SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale') pada y yang distandardisasi dengan
                   mean/std train (log1p dulu bila target log) = skala standar-log; prediksi & metrik di skala itu.
       klasifikasi SVC(kernel='rbf', C=1.0, gamma='scale', probability=True); keluaran label kelas asli + peluang.
  4. VAL hanya untuk ambang F1 & temperatur (klasifikasi), sama dengan baseline lain. Laporan di TEST.
  5. Metrik sama dengan metode usulan: regresi RMSE/MAE/R2 (skala standar-log; CRPS = NaN karena SVR hanya
     memberi prediksi titik), klasifikasi metrics_cls (AUC, F1 argmax = sama dengan usulan, Acc, Brier, ...;
     F1 dengan ambang val disimpan sebagai f1_macro_thr / keterangan "[ambang val ...]" di result.txt).

KELUARAN  (results/twostage_{imputer}_svm/{dataset}/MAR{rr}/seed{k}/)
  twostage_{imputer}_svm_{val,test}.csv : id, y_asli, y_pred (nilai asli) [+ y_pred_ambang, p_kelas]
  twostage_{imputer}_svm_metrics.json   : metrik + mutu imputasi (disalin dari tahap 1) + setting
  result.txt                            : ringkasan (mode append)

Pemakaian (dari folder baselines_eksternal, SETELAH run_imputasi.py):
  python run_svm.py --imputer mean --dataset bike --ratio 0.3 --seed 0
  python run_svm.py --imputer knn --dataset shoppers --ratio 0.3 --seed 0
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = next((d for d in (_HERE, os.path.dirname(_HERE)) if os.path.isfile(os.path.join(d, 'data_loader.py'))), None)
if PROJECT_DIR is None:
    raise FileNotFoundError('data_loader.py tidak ditemukan di folder skrip maupun folder induknya')
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from data_loader import load_split, y_true_index, y_true_z, load_ids                  # noqa: E402
import metrics_cls as mc                                                     # noqa: E402

IMPUTERS = ['mean', 'knn', 'mice', 'missforest', 'hyperimpute', 'miwae', 'diffputer']
PARTS = ['train', 'val', 'test']


def svm_features(frames, ref_df):
    """one-hot drop_first untuk kolom kategorik (dari dtype data asli) + StandardScaler fit di train."""
    from sklearn.preprocessing import StandardScaler
    feat = list(frames['train'].columns)
    is_num = {c: pd.api.types.is_numeric_dtype(ref_df[c]) and not pd.api.types.is_bool_dtype(ref_df[c]) for c in feat}
    cat_cols = [c for c in feat if not is_num[c]]
    num_cols = [c for c in feat if is_num[c]]
    D = {}
    for p, f in frames.items():
        f = f.copy()
        for c in cat_cols:
            f[c] = f[c].astype(str)
        D[p] = pd.get_dummies(f, columns=cat_cols, drop_first=True, dtype=float) if cat_cols else f
    for p in D:
        if p != 'train':
            D['train'], D[p] = D['train'].align(D[p], join='left', axis=1, fill_value=0)
    sc = StandardScaler().fit(D['train'].to_numpy(np.float64))
    return {p: sc.transform(D[p].to_numpy(np.float64)) for p in D}, cat_cols, num_cols, list(D['train'].columns)


def main():
    ap = argparse.ArgumentParser(description='Tahap 2 baseline dua tahap: SVM pada data hasil imputasi')
    ap.add_argument('--imputer', required=True, choices=IMPUTERS)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--ratio', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mech', default='MAR')
    ap.add_argument('--out_dir', default=os.path.join(_HERE, 'results_zlog'),
                    help='folder hasil SVM (bawaan: results_zlog/)')
    ap.add_argument('--imp_dir', default=os.path.join(_HERE, 'results'),
                    help='folder induk hasil imputasi tahap 1 (bawaan: results_zlog/; imputasi tidak memakai y)')
    ap.add_argument('--thr_objective', default='f1_macro', choices=['f1_macro', 'f1_weighted'])
    a = ap.parse_args()

    t_all = time.time()
    data, meta = load_split(a.dataset, a.seed, a.ratio, a.mech)
    task, target = meta['task'], meta['target']
    feat = list(meta['num_cols']) + list(meta['cat_cols'])
    tag = f'{a.mech}{int(round(a.ratio * 100)):02d}'
    imp_dir = f'{a.imp_dir}/imputasi/{a.imputer}/{a.dataset}/{tag}/seed{a.seed}'
    if not all(os.path.exists(f'{imp_dir}/imputed_{p}.csv') for p in PARTS):
        cmd = (f'python run_imputasi_diffputer.py --dataset {a.dataset} --seed {a.seed}' if a.imputer == 'diffputer'
               else f'python run_imputasi.py --imputer {a.imputer} --dataset {a.dataset} --seed {a.seed}')
        raise FileNotFoundError(f'Hasil imputasi belum ada di {imp_dir}. Jalankan dulu:\n  {cmd}')
    method = f'twostage_{a.imputer}_svm'
    out_dir = f'{a.out_dir}/{method}/{a.dataset}/{tag}/seed{a.seed}'
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== {a.imputer} -> {"SVC" if task == "classification" else "SVR"} | {a.dataset} ({task}) | {tag} '
          f'seed{a.seed} ===', flush=True)

    ref = data['train']['df']
    frames = {}
    for p in PARTS:
        f = pd.read_csv(f'{imp_dir}/imputed_{p}.csv', dtype={c: ref[c].dtype for c in feat
                                                             if not pd.api.types.is_numeric_dtype(ref[c])})
        assert len(f) == len(data[p]['df']), f'jumlah baris {p} tidak cocok dengan split'
        assert np.array_equal(f[target].astype(str).to_numpy(), data[p]['df'][target].astype(str).to_numpy()), \
            f'kolom target {p} tidak cocok dengan split (seed/rasio berbeda?)'
        assert not f[feat].isna().any().any(), f'masih ada sel kosong di imputed_{p}.csv'
        # pola missing tahap 1 harus SAMA dengan mask data sekarang (split/mask bisa berubah antar versi data)
        mp = f'{imp_dir}/missing_{p}.csv'
        if os.path.exists(mp):
            miss_old = pd.read_csv(mp)[feat].isna().to_numpy()
            if miss_old.shape != data[p]['mask_feat'].shape or not np.array_equal(miss_old, np.asarray(data[p]['mask_feat'], bool)):
                raise SystemExit(f'Pola missing di {mp} BERBEDA dengan mask data sekarang (seed/mask lain). '
                                 f'Jalankan ulang tahap 1 untuk data ini:\n  ' +
                                 (f'python run_imputasi_diffputer.py --dataset {a.dataset} --seed {a.seed}' if a.imputer == 'diffputer'
                                  else f'python run_imputasi.py --imputer {a.imputer} --dataset {a.dataset} --seed {a.seed} --ulang'))
        else:
            print(f'[PERINGATAN] {mp} tidak ada: pola missing tahap 1 tidak bisa dicek')
        frames[p] = f[feat]
    Z, cat_cols, num_cols, cols = svm_features(frames, ref)

    from sklearn.svm import SVC, SVR
    summary = {'method': method, 'imputer': a.imputer, 'prediktor': 'SVC' if task == 'classification' else 'SVR',
               'task': task, 'dataset': a.dataset, 'mech': a.mech, 'ratio': a.ratio, 'seed': a.seed, 'args': vars(a)}
    t0 = time.time()
    if task == 'classification':
        Y = {p: y_true_index(data[p]['df'], meta) for p in PARTS}
        model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=a.seed)
        model.fit(Z['train'], Y['train'])
        proba = {p: model.predict_proba(Z[p]) for p in ('val', 'test')}
        T_ = mc.fit_temperature(proba['val'], Y['val'], objective='brier')
        thr = mc.fit_threshold(proba['val'], Y['val'], objective=a.thr_objective)
        summary.update({'T': T_, 'ambang': thr})
        for p in ('val', 'test'):
            m_raw = mc.cls_metrics(proba[p], Y[p], thr)
            summary[p] = {'y_mentah': m_raw, 'y_ts': mc.cls_metrics(mc.apply_temperature(proba[p], T_), Y[p], thr)}
            print(f'Prediksi y ({p}) → {mc.fmt(m_raw)}{" [ambang dipilih di val ini]" if p == "val" else ""}')
    else:
        yo = {p: y_true_z(data[p]['df'], meta) for p in PARTS}                         # skala standar-log
        mu, sd = float(meta['mean'][meta['y_idx']]), float(meta['std'][meta['y_idx']])
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        model.fit(Z['train'], yo['train'])                     # = (log1p(y) - mu) / sd seperti versi lama
        point = {p: model.predict(Z[p]) for p in ('val', 'test')}                       # skala standar-log
        for p in ('val', 'test'):
            y, pt = yo[p], point[p]
            m = {'rmse': float(np.sqrt(((y - pt) ** 2).mean())), 'mae': float(np.abs(y - pt).mean()),
                 'r2': float(1 - ((y - pt) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 'crps': float('nan')}
            summary[p] = m
            print(f'Prediksi y ({p}) → RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}  R2={m["r2"]:.4f}  (CRPS: tidak ada, SVR titik)')
    summary['svm_time_s'] = time.time() - t0
    summary['svm'] = {'params': model.get_params(), 'kolom_onehot': cat_cols, 'kolom_numerik': num_cols,
                      'n_fitur_model': len(cols), 'standardisasi_y': None if task == 'classification'
                      else {'mean': float(mu), 'std': float(sd), 'log1p': bool(meta['log_target'])}}
    with open(f'{imp_dir}/imputasi_metrics.json') as f:                    # mutu imputasi dari tahap 1
        im = json.load(f)
    summary['imputasi'] = {p: im.get(p) for p in PARTS}

    for p in ('val', 'test'):
        idx_path = f'datasets/{a.dataset}/splits/seed{a.seed}/{p}_idx.npy'
        out = pd.DataFrame({'id': load_ids(idx_path, len(data[p]['df']))})
        out['y_asli'] = data[p]['df'][target].values if task == 'classification' else yo[p]
        if task == 'classification':
            cl = meta['classes']
            out['y_pred'] = np.array(cl, dtype=object)[proba[p].argmax(1)]
            out['y_pred_ambang'] = np.array(cl, dtype=object)[mc.pred_at(proba[p], summary['ambang'])]
            for k, c in enumerate(cl):
                out[f'p_{c}'] = proba[p][:, k]
        else:
            out['y_pred'] = point[p]
        out.to_csv(f'{out_dir}/{method}_{p}.csv', index=False)
    summary['total_time_s'] = time.time() - t_all
    with open(f'{out_dir}/{method}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    te = summary['test']
    # Klasifikasi: Acc & F1(macro) = ARGMAX (ambang 0.5), SAMA dengan metode usulan (cls_standar). F1 dengan ambang
    # dari val dicetak terpisah sebagai keterangan "[ambang val ...]" (versi lama mencetak F1 ambang sebagai F1 utama).
    ym = te.get('y_mentah', {})                                         # regresi: tidak ada y_mentah
    thr_txt = (f" [ambang val {summary['ambang']:.3f}: Acc {ym['acc_thr']:.4f} F1(macro) {ym['f1_macro_thr']:.4f}]"
               if task == 'classification' and 'f1_macro_thr' in ym else '')
    line = (f"test | {method} | " + (f"Acc {ym['acc']:.4f} F1(macro) {ym['f1_macro']:.4f} "
                                     f"AUC {ym['auc']:.4f} Brier {ym['brier']:.4f}{thr_txt}"
                                     if task == 'classification' else
                                     f"RMSE {te['rmse']:.4f} MAE {te['mae']:.4f} R2 {te['r2']:.4f}")
            + f" | waktu {summary['total_time_s']:.0f}s\n")
    with open(f'{out_dir}/result.txt', 'a') as f:
        f.write(line)
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {out_dir}/')


if __name__ == '__main__':
    main()