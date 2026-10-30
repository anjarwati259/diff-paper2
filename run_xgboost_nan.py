"""
Baseline: XGBoost dengan penanganan NaN bawaan (tanpa imputasi), hyperparameter DEFAULT (TANPA tuning)
+ early stopping di val (praktik standar XGBoost).

- Split, mask, dan data diambil dari data_loader.py (sama persis dengan DiffPuter & usulan).
- Fitur yang di-mask diberi NaN; XGBoost belajar arah default tiap split untuk nilai hilang
  (sparsity-aware split finding, Chen & Guestrin, KDD 2016).
- Kategorik: tipe category pandas (enable_categorical=True), NaN juga ditangani bawaan.
- Target: log1p bila dataset memakai log target (sama dengan data_loader), dikembalikan ke skala asli.
- Hyperparameter: DEFAULT XGBoost (sama dengan XGBRegressor bawaan):
    learning_rate (eta) 0.3, max_depth 6, min_child_weight 1, subsample 1, colsample_bytree 1,
    reg_lambda 1, tree_method hist. TIDAK ada grid search.
- Jumlah pohon: early stopping di val (maks --max_rounds 5000, berhenti bila RMSE val tidak membaik
  selama --early_stop 100 ronde); prediksi memakai iterasi terbaik. Model dilatih di train saja.
- Prediksi titik: model reg:squarederror.
- Interval: model kuantil (reg:quantileerror, XGBoost >= 2.0) dengan hyperparameter default yang sama,
  pada level k/(N+1), k = 1..N (N = --num_q, default 10 = num_trials DiffPuter).
  Kuantil 0.05 & 0.95 juga dilatih langsung (coverage_q05_q95).
- CRPS: kuantil k/(N+1) BUKAN sampel acak, sehingga rumus "fair" (pembagi 2N(N-1), untuk sampel iid)
  meremehkan CRPS ~7.5%. Untuk kuantil dipakai rumus biasa (pembagi 2N^2) -> kunci 'crps'.
  Nilai versi fair tetap disimpan sebagai 'crps_fair_kuantil'.
- 'coverage_p5_p95' dari 10 kuantil nominalnya hanya ~0.75; untuk membandingkan interval dengan
  DiffPuter pakai coverage min-max (nominal 0.818 bila N = 10) dan coverage_q05_q95.

KLASIFIKASI: bila Info JSON dataset berisi "task_type": "classification" (mis. shoppers), skrip otomatis
memakai XGBoost classifier (run_cls). Default --setting sklearn_default = SAMA dengan model_classification.py
(XGBClassifier default 100 pohon tanpa early stopping, kategorik dari dtype -> one-hot drop_first,
StandardScaler, 5-fold CV di train, metrik weighted + confusion matrix); bedanya hanya: sel yang di-mask
tetap NaN (tanpa imputasi). --setting es = early stopping di val. Metrik dari metrics_cls.py.

Contoh:
    python run_xgboost_nan.py --dataset shoppers --ratio 0.3 --seed 0     # klasifikasi (otomatis)
    python run_xgboost_nan.py --dataset bike --ratio 0.3 --seed 0
    python run_xgboost_nan.py --dataset bike --ratio 0.3 --seed 0 --device cuda
Hasil (folder terpisah dari versi tuning lama):
    results/xgboost_default/{dataset}/MAR30/seed0/xgboost_default_{val,test}.csv
    results/xgboost_default/{dataset}/MAR30/seed0/xgboost_default_{val,test}_samples.npy   [n, N] skala asli
    results/xgboost_default/{dataset}/MAR30/seed0/xgboost_default_metrics.json
    results/xgboost_default/{dataset}/MAR30/seed0/result.txt  (append)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from data_loader import load_split


# ----------------------------------------------------------------------
# Metrik y (salinan fungsi metrik di run_diffputer.py, supaya identik)
# ----------------------------------------------------------------------
def crps_fair(samples, y):
    N = samples.shape[0]
    t1 = np.abs(samples - y[None, :]).mean(0)
    t2 = np.abs(samples[:, None, :] - samples[None, :, :]).sum((0, 1)) / (2 * N * (N - 1))
    return float((t1 - t2).mean())


def crps_quantile(Q, y):
    """Q [N, n] kuantil level k/(N+1). CRPS distribusi empiris titik-titik kuantil (pembagi 2N^2)."""
    N = Q.shape[0]
    t1 = np.abs(Q - y[None, :]).mean(0)
    t2 = np.abs(Q[:, None, :] - Q[None, :, :]).sum((0, 1)) / (2 * N * N)
    return float((t1 - t2).mean())


def y_metrics(y_true, samples, y_point=None):
    """samples: [N, n] skala asli. y_point: prediksi titik (default rata-rata sampel)."""
    y_pred = samples.mean(0) if y_point is None else y_point
    lo, hi = np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)
    mn, mx = samples.min(0), samples.max(0)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    rank = (samples < y_true[None, :]).sum(0)
    hist = np.bincount(rank, minlength=samples.shape[0] + 1) / len(y_true)
    return {
        'rmse': float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        'mae': float(np.abs(y_true - y_pred).mean()),
        'r2': float(1 - ss_res / ss_tot),
        'crps': crps_fair(samples, y_true),
        'coverage_p5_p95': float(((y_true >= lo) & (y_true <= hi)).mean()),
        'width_p5_p95': float((hi - lo).mean()),
        'coverage_minmax': float(((y_true >= mn) & (y_true <= mx)).mean()),
        'coverage_minmax_ideal': (samples.shape[0] - 1) / (samples.shape[0] + 1),
        'width_minmax': float((mx - mn).mean()),
        'rank_hist_dev': float(np.abs(hist - 1 / len(hist)).mean()),
    }


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def make_frames(data, meta):
    """Fitur skala asli; sel yang di-mask -> NaN; kategorik -> category (kategori dari train)."""
    num_cols, cat_cols = list(meta['num_cols']), list(meta['cat_cols'])
    feat_cols = num_cols + cat_cols
    frames = {}
    for part, d in data.items():
        X = d['df'][feat_cols].copy()
        X = X.mask(np.asarray(d['mask_feat'], dtype=bool))      # urutan mask_feat = num -> kat
        frames[part] = X
    for c in cat_cols:
        cats = pd.Series(data['train']['df'][c]).dropna().unique()
        cats = sorted(cats.tolist(), key=str)
        for part in frames:
            frames[part][c] = pd.Categorical(frames[part][c], categories=cats)
    for c in num_cols:
        for part in frames:
            frames[part][c] = frames[part][c].astype(np.float64)
    return frames, feat_cols


def to_model_space(y, log_target):
    return np.log1p(y) if log_target else y


def to_orig(y, log_target):
    return np.expm1(y) if log_target else y


# ----------------------------------------------------------------------
# Model — hyperparameter DEFAULT XGBoost, tanpa tuning, tanpa early stopping
# ----------------------------------------------------------------------
def default_params(args):
    return dict(tree_method='hist', device=args.device, seed=args.seed, nthread=args.nthread,
                eta=0.3, max_depth=6, min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                reg_lambda=1.0)


def fit_point(dtr, dva, args):
    p = {**default_params(args), 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    return xgb.train(p, dtr, num_boost_round=args.max_rounds, evals=[(dva, 'val')],
                     early_stopping_rounds=args.early_stop, verbose_eval=False)


def fit_quantile(dtr, dva, alphas, args):
    p = {**default_params(args), 'objective': 'reg:quantileerror', 'quantile_alpha': np.asarray(alphas)}
    return xgb.train(p, dtr, num_boost_round=args.max_rounds, evals=[(dva, 'val')],
                     early_stopping_rounds=args.early_stop, verbose_eval=False)


def predict_best(bst, dm):
    return bst.predict(dm, iteration_range=(0, bst.best_iteration + 1))


# ----------------------------------------------------------------------
# KLASIFIKASI (task_type = classification di Info JSON) — dipilih otomatis oleh main()
# ----------------------------------------------------------------------
def prepare_cls_frames(data, meta, target):
    """Persiapan fitur SAMA dengan prepare_data() di model_classification.py:
      - kolom kategorik dideteksi dari DTYPE (object/category/bool), BUKAN dari Info JSON
        (di Shoppers: Month & VisitorType; kolom berkode angka seperti Browser dianggap numerik)
      - one-hot pd.get_dummies(drop_first=True), kolom test disamakan ke train (align join='left', isi 0)
      - StandardScaler di-fit di train untuk kolom numerik
    Satu-satunya beda (disengaja): sel yang di-mask TETAP NaN (tidak diimputasi), karena baseline ini menguji
    penanganan NaN bawaan XGBoost. StandardScaler sklearn mengabaikan NaN saat fit dan mempertahankannya.
    Untuk kategorik yang hilang, semua kolom dummy-nya diberi NaN (kalau dibiarkan 0, baris itu terbaca
    sebagai kategori acuan yang di-drop)."""
    from sklearn.preprocessing import StandardScaler
    feat_cols = list(meta['num_cols']) + list(meta['cat_cols'])
    raw = {part: d['df'][feat_cols].mask(np.asarray(d['mask_feat'], dtype=bool)) for part, d in data.items()}
    # dtype dideteksi dari data train asli (sebelum masking, karena masking bisa mengubah int -> float)
    ref = data['train']['df'][feat_cols]
    # = select_dtypes(include=['object','category','bool']) / include=[np.number] (tanpa peringatan pandas 3)
    is_num = {c: pd.api.types.is_numeric_dtype(ref[c]) and not pd.api.types.is_bool_dtype(ref[c]) for c in feat_cols}
    cat_cols = [c for c in feat_cols if not is_num[c]]
    num_cols = [c for c in feat_cols if is_num[c]]
    frames = {}
    for part, X in raw.items():
        miss_cat = {c: X[c].isna().to_numpy() for c in cat_cols}
        D = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float) if cat_cols else X.copy()
        for c in cat_cols:                                    # kategori hilang -> dummy NaN
            dcols = [k for k in D.columns if k.startswith(f'{c}_')]
            D.loc[miss_cat[c], dcols] = np.nan
        frames[part] = D
    for part in frames:
        if part != 'train':
            frames['train'], frames[part] = frames['train'].align(frames[part], join='left', axis=1, fill_value=0)
    if num_cols:
        sc = StandardScaler().fit(frames['train'][num_cols])
        for part in frames:
            frames[part][num_cols] = sc.transform(frames[part][num_cols])
    return frames, cat_cols, num_cols


def run_cls(args, data, meta, t_all):
    """XGBoost classifier untuk KLASIFIKASI, pengaturan SAMA dengan model_classification.py:
      XGBClassifier(eval_metric='logloss') hyperparameter default sklearn-API xgboost:
        n_estimators 100 (TETAP, tanpa early stopping), learning_rate 0.3, max_depth 6, tree_method hist.
      Fit di seluruh train; 5-fold StratifiedKFold CV (shuffle, random_state 42) di train sebagai 'validasi_cv'.
      Metrik utama skrip lama: accuracy, precision/recall/F1 weighted, ROC-AUC (biner: p kelas 1; multikelas:
      OvR weighted) + confusion matrix & classification report.
    Tambahan (tidak mengubah model): macro-F1, NLL, Brier, ECE, AURC (metrics_cls.py, sama dengan usulan),
    evaluasi di set val kita, temperature scaling (T dari val, objektif Brier), dan AMBANG F1 dari val
    (biner; aturan sama dengan metode usulan, metrik *_thr).
    --setting es = versi lama Paper 2 (xgb.train + early stopping logloss di val)."""
    import metrics_cls as mc
    from data_loader import y_true_index
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix, classification_report

    K = len(meta['classes'])
    y = {part: y_true_index(data[part]['df'], meta) for part in data}
    tag = f'{args.mech}{int(round(args.ratio * 100))}'
    varian = args.varian if args.setting == 'sklearn_default' else f'{args.varian}_es'
    pred_dir = f'{args.out_dir}/{varian}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)

    if args.setting == 'sklearn_default':
        frames, cat_cols, num_cols = prepare_cls_frames(data, meta, meta['target'])
        info = (f'XGBClassifier default (n_estimators 100 tetap, eta 0.3, max_depth 6) | kategorik (dtype): {cat_cols} '
                f'-> one-hot drop_first | numerik: {len(num_cols)} kolom, StandardScaler | NaN dipertahankan')

        def make_model():
            return xgb.XGBClassifier(eval_metric='logloss', device=args.device,
                                     n_jobs=None if args.nthread < 0 else args.nthread)
    else:
        frames, _ = make_frames(data, meta)
        info = 'xgb.train default (eta 0.3, max_depth 6) + early stopping logloss di val, kategorik native'
    print(f'=== {varian} (KLASIFIKASI, {K} kelas {meta["classes"]}) | {args.dataset} {tag} seed{args.seed} | '
          f'xgboost {xgb.__version__} | missing train {data["train"]["mask_feat"].mean():.3f} ===')
    print(info)

    summary = {'varian': varian, 'task': 'classification', 'setting': args.setting, 'dataset': args.dataset,
               'mech': args.mech, 'ratio': args.ratio, 'seed': args.seed, 'classes': meta['classes'],
               'xgboost_version': xgb.__version__, 'info': info}
    t0 = time.time()
    if args.setting == 'sklearn_default':
        # ---- validasi: 5-fold CV di train (sama dengan cross_validate_model) ----
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        folds = []
        Xtr = frames['train']
        for k, (a_idx, b_idx) in enumerate(skf.split(Xtr, y['train']), start=1):
            mdl = make_model().fit(Xtr.iloc[a_idx], y['train'][a_idx])
            m = mc.cls_metrics(mdl.predict_proba(Xtr.iloc[b_idx]), y['train'][b_idx])
            folds.append(m)
            print(f'  Fold {k}: acc={m["acc"]:.4f}, prec(w)={m["precision_w"]:.4f}, rec(w)={m["recall_w"]:.4f}, '
                  f'f1(w)={m["f1_weighted"]:.4f}, roc_auc={m["auc_weighted"]:.4f}')
        cv = {f'{k}_mean': float(np.mean([f[k] for f in folds])) for k in folds[0]}
        cv.update({f'{k}_std': float(np.std([f[k] for f in folds], ddof=1)) for k in folds[0]})
        summary['validasi_cv'] = cv
        print(f'Rata-rata {args.cv}-fold CV (train): acc={cv["acc_mean"]:.4f}, f1(w)={cv["f1_weighted_mean"]:.4f}, '
              f'f1(macro)={cv["f1_macro_mean"]:.4f}, roc_auc={cv["auc_weighted_mean"]:.4f}')
        # ---- model final di seluruh train ----
        model = make_model().fit(Xtr, y['train'])
        P = {part: model.predict_proba(frames[part]) for part in ('val', 'test')}
        summary['n_trees'] = model.get_booster().num_boosted_rounds()
        summary['params'] = {k: v for k, v in model.get_xgb_params().items() if v is not None}
    else:
        dm = {part: xgb.DMatrix(frames[part], label=y[part], enable_categorical=True, missing=np.nan)
              for part in data}
        obj = {'objective': 'binary:logistic', 'eval_metric': 'logloss'} if K == 2 else \
              {'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'num_class': K}
        params = {**default_params(args), **obj}
        bst = xgb.train(params, dm['train'], num_boost_round=args.max_rounds, evals=[(dm['val'], 'val')],
                        early_stopping_rounds=args.early_stop, verbose_eval=False)

        def proba(part):
            pr = np.asarray(predict_best(bst, dm[part]))
            return np.stack([1 - pr, pr], axis=1) if K == 2 else pr.reshape(len(y[part]), K)
        P = {part: proba(part) for part in ('val', 'test')}
        summary['n_trees'] = bst.best_iteration + 1
        summary['params'] = {k: v for k, v in params.items() if k not in ('device', 'nthread')}
    summary['train_time_s'] = time.time() - t0
    T = mc.fit_temperature(P['val'], y['val'], objective='brier')     # sama dengan metode usulan
    thr = mc.fit_threshold(P['val'], y['val'], objective=args.thr_objective)   # ambang F1 dari val (biner)
    summary['T'] = T
    summary['ambang'] = thr

    for part in ('val', 'test'):
        m_raw = mc.cls_metrics(P[part], y[part], thr)
        P_ts = mc.apply_temperature(P[part], T)
        m_ts = mc.cls_metrics(P_ts, y[part], thr)
        summary[part] = {'y_mentah': m_raw, 'y_ts': m_ts}
        print(f'Prediksi y ({part}, mentah) → {mc.fmt(m_raw)}{" [ambang dipilih di val ini]" if part == "val" else ""}')
        print(f'Prediksi y ({part}, TS T={T:.3f}) → NLL={m_ts["nll"]:.4f}  Brier={m_ts["brier"]:.4f}  '
              f'ECE={m_ts["ece"]:.4f}{" [T dipas di val ini, optimistis]" if part == "val" else ""}')
        if part == 'test':
            pred = P[part].argmax(1)
            print('Confusion Matrix (test):')
            print(confusion_matrix(y[part], pred, labels=list(range(K))))
            print('Classification Report (test):')
            print(classification_report(y[part], pred, labels=list(range(K)),
                                        target_names=[str(c) for c in meta['classes']], zero_division=0))

        idx_path = f'datasets/{args.dataset}/splits/seed{args.seed}/{part}_idx.npy'
        out = pd.DataFrame({'id': np.load(idx_path) if os.path.exists(idx_path) else np.arange(len(y[part]))})
        out['y_asli'] = data[part]['df'][meta['target']].values
        out['y_pred'] = np.array(meta['classes'], dtype=object)[P[part].argmax(1)]
        if thr is not None:
            out['y_pred_ambang'] = np.array(meta['classes'], dtype=object)[mc.pred_at(P[part], thr)]
        for k, cl in enumerate(meta['classes']):
            out[f'p_{cl}'] = P[part][:, k]
        for k, cl in enumerate(meta['classes']):
            out[f'p_ts_{cl}'] = P_ts[:, k]
        out.to_csv(f'{pred_dir}/{varian}_{part}.csv', index=False)

        with open(f'{pred_dir}/result.txt', 'a') as f:
            f.write(f'{part} | Acc {m_raw["acc"]:.4f} Prec(w) {m_raw["precision_w"]:.4f} Rec(w) {m_raw["recall_w"]:.4f} '
                    f'F1(w) {m_raw["f1_weighted"]:.4f} F1(macro) {m_raw["f1_macro"]:.4f} AUC {m_raw["auc"]:.4f} '
                    f'AUC(w) {m_raw["auc_weighted"]:.4f} NLL {m_raw["nll"]:.4f} Brier {m_raw["brier"]:.4f} '
                    f'ECE {m_raw["ece"]:.4f}'
                    + (f' | ambang {thr:.3f}: Acc {m_raw["acc_thr"]:.4f} F1(macro) {m_raw["f1_macro_thr"]:.4f} '
                       f'F1(w) {m_raw["f1_weighted_thr"]:.4f}' if thr is not None else '')
                    + f' | TS NLL {m_ts["nll"]:.4f} Brier {m_ts["brier"]:.4f} '
                    f'ECE {m_ts["ece"]:.4f} | T {T:.3f} | pohon {summary["n_trees"]}\n')

    summary['total_time_s'] = time.time() - t_all
    with open(f'{pred_dir}/{varian}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'Jumlah pohon: {summary["n_trees"]}')
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{varian}_*')


# ----------------------------------------------------------------------
# Utama
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='Baseline XGBoost NaN bawaan, hyperparameter default')
    p.add_argument('--dataset', required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mech', default='MAR')
    p.add_argument('--varian', default='xgboost_default')
    p.add_argument('--out_dir', default='results')
    p.add_argument('--device', default='cpu', help='cpu atau cuda')
    p.add_argument('--nthread', type=int, default=-1)
    p.add_argument('--num_q', type=int, default=10, help='jumlah kuantil = "sampel" (N)')
    p.add_argument('--max_rounds', type=int, default=5000, help='maks. jumlah pohon')
    p.add_argument('--early_stop', type=int, default=100, help='ronde early stopping di val')
    # khusus klasifikasi
    p.add_argument('--setting', default='sklearn_default', choices=['sklearn_default', 'es'],
                   help="klasifikasi: 'sklearn_default' = sama dengan model_classification.py (XGBClassifier "
                        "default, 100 pohon, one-hot drop_first, StandardScaler); 'es' = early stopping di val")
    p.add_argument('--cv', type=int, default=5, help='klasifikasi: jumlah fold CV di train (validasi)')
    p.add_argument('--thr_objective', default='f1_macro', choices=['f1_macro', 'f1_weighted'],
                   help='klasifikasi biner: F1 yang dimaksimalkan saat memilih ambang di val (sama dengan usulan)')
    args = p.parse_args()

    t_all = time.time()
    data, meta = load_split(args.dataset, args.seed, args.ratio, args.mech)
    if meta.get('task', 'regression') == 'classification':
        return run_cls(args, data, meta, t_all)                 # XGBoost classifier (lihat run_cls)
    log_t = bool(meta.get('log_target', False))
    target = meta['target']
    frames, feat_cols = make_frames(data, meta)

    y = {part: data[part]['df'][target].to_numpy(dtype=np.float64) for part in data}
    dm = {part: xgb.DMatrix(frames[part], label=to_model_space(y[part], log_t),
                            enable_categorical=True, missing=np.nan) for part in data}

    tag = f'{args.mech}{int(round(args.ratio * 100))}'
    pred_dir = f'{args.out_dir}/{args.varian}/{args.dataset}/{tag}/seed{args.seed}'
    os.makedirs(pred_dir, exist_ok=True)
    print(f'=== {args.varian} | {args.dataset} {tag} seed{args.seed} | xgboost {xgb.__version__} | '
          f'{len(feat_cols)} fitur | missing train {data["train"]["mask_feat"].mean():.3f} ===')
    print(f'Hyperparameter DEFAULT (tanpa tuning): eta 0.3, max_depth 6, min_child_weight 1, subsample 1, '
          f'colsample 1 | early stopping di val ({args.early_stop} ronde, maks {args.max_rounds} pohon)')

    # ---- model titik & kuantil, dilatih di train saja ----
    t0 = time.time()
    bst_point = fit_point(dm['train'], dm['val'], args)
    t_point = time.time() - t0
    N = args.num_q
    alphas = [k / (N + 1) for k in range(1, N + 1)]
    t0 = time.time()
    bst_q = fit_quantile(dm['train'], dm['val'], alphas, args)
    bst_q90 = fit_quantile(dm['train'], dm['val'], [0.05, 0.95], args)
    t_q = time.time() - t0

    summary = {'varian': args.varian, 'dataset': args.dataset, 'mech': args.mech,
               'ratio': args.ratio, 'seed': args.seed, 'xgboost_version': xgb.__version__,
               'params': {k: v for k, v in default_params(args).items() if k not in ('device', 'nthread')},
               'n_trees_point': bst_point.best_iteration + 1, 'n_trees_quantile': bst_q.best_iteration + 1,
               'n_trees_q05_q95': bst_q90.best_iteration + 1,
               'tuning': 'tidak ada (default) + early stopping di val',
               'quantile_levels': alphas, 'train_time_point_s': t_point, 'train_time_quantile_s': t_q}

    for part in ('val', 'test'):
        y_point = to_orig(predict_best(bst_point, dm[part]), log_t)
        Q = np.sort(np.asarray(predict_best(bst_q, dm[part])).reshape(len(y[part]), -1), axis=1)  # perbaiki crossing
        samples = to_orig(Q, log_t).T                        # [N, n] skala asli
        Q90 = np.sort(np.asarray(predict_best(bst_q90, dm[part])).reshape(len(y[part]), -1), axis=1)
        lo90, hi90 = to_orig(Q90[:, 0], log_t), to_orig(Q90[:, 1], log_t)

        m = y_metrics(y[part], samples, y_point=y_point)
        m['crps_fair_kuantil'] = m['crps']                   # rumus sampel iid (meremehkan untuk kuantil)
        m['crps'] = crps_quantile(samples, y[part])          # rumus untuk kuantil (dipakai membandingkan)
        m['rmse_quantile_mean'] = float(np.sqrt(((y[part] - samples.mean(0)) ** 2).mean()))
        m['coverage_q05_q95'] = float(((y[part] >= lo90) & (y[part] <= hi90)).mean())
        m['width_q05_q95'] = float((hi90 - lo90).mean())
        summary[part] = m
        print(f'Prediksi y ({part}) → RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}  R2={m["r2"]:.4f}  '
              f'CRPS={m["crps"]:.4f}  cov p5-95={m["coverage_p5_p95"]:.3f}  '
              f'cov q05-q95={m["coverage_q05_q95"]:.3f}  '
              f'cov min-max={m["coverage_minmax"]:.3f} (ideal {m["coverage_minmax_ideal"]:.3f})')

        idx_path = f'datasets/{args.dataset}/splits/seed{args.seed}/{part}_idx.npy'
        out = pd.DataFrame({'id': np.load(idx_path) if os.path.exists(idx_path)
                            else np.arange(len(y[part]))})
        for c in feat_cols:
            out[c] = frames[part][c].astype(object).values
        out['y_asli'] = y[part]
        out['y_pred'] = y_point
        out['y_low'] = lo90
        out['y_high'] = hi90
        out.to_csv(f'{pred_dir}/{args.varian}_{part}.csv', index=False)
        np.save(f'{pred_dir}/{args.varian}_{part}_samples.npy', samples.T.astype(np.float32))

        with open(f'{pred_dir}/result.txt', 'a') as f:
            f.write(f'{part} | y RMSE {m["rmse"]:.4f} MAE {m["mae"]:.4f} R2 {m["r2"]:.4f} '
                    f'CRPS {m["crps"]:.4f} cov {m["coverage_p5_p95"]:.3f} '
                    f'cov_q {m["coverage_q05_q95"]:.3f} covmm {m["coverage_minmax"]:.3f}\n')

    summary['total_time_s'] = time.time() - t_all
    with open(f'{pred_dir}/{args.varian}_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Jumlah pohon (early stopping): titik {summary["n_trees_point"]}, kuantil {summary["n_trees_quantile"]}, '
          f'q05/q95 {summary["n_trees_q05_q95"]}')
    print(f'Selesai ({summary["total_time_s"]:.0f}s). Hasil: {pred_dir}/{args.varian}_*')


if __name__ == '__main__':
    main()