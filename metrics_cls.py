"""
Metrik dan pembacaan probabilitas untuk target KLASIFIKASI (dipakai metode usulan & baseline,
supaya rumusnya identik di semua metode).

Probabilitas dari N sampel diffusion (blok one-hot +-0.5):
  q_k   = rata-rata sampel kolom k, digeser ke [0, 1] (q = rata-rata + 0.5), di-clip ke [eps, 1],
          lalu dinormalisasi supaya jumlahnya 1  -> probabilitas "mentah" (tanpa parameter)
  p_k(T) = q_k^(1/T) / sum_j q_j^(1/T)  = softmax(log q / T)  -> temperature scaling
          (Guo dkk., ICML 2017). T dipilih di data VALIDASI dengan meminimalkan NLL, lalu dipakai di test.
Prediksi kelas = argmax p (argmax tidak berubah oleh T).

Metrik (semua dari vektor probabilitas P [n, K] dan label y [n] berindeks 0..K-1):
  acc, balanced_acc, f1_macro, auc (biner: kelas 1; multikelas: one-vs-rest macro),
  nll (log loss), brier (versi asli Brier 1950: jumlah atas K kelas, rata-rata baris;
  untuk biner = 2 x brier_score_loss sklearn -> 'brier_biner' juga disimpan),
  ece (15 bin, berdasarkan confidence = max p), aurc (area under risk-coverage, selective
  prediction; Geifman & El-Yaniv 2017; makin kecil makin baik), entropi rata-rata.
Versi 'weighted' (sama dengan model_classification.py: precision/recall/F1 average='weighted', AUC multikelas
  OvR weighted): precision_w, recall_w, f1_weighted, auc_weighted (biner: = auc).
  Catatan: weighted ditimbang jumlah sampel tiap kelas -> pada data tidak seimbang didominasi kelas mayoritas
  (nilainya dekat akurasi); macro memberi bobot sama ke kelas minoritas.
Ambang keputusan (biner): fit_threshold memilih ambang p(kelas 1) yang memaksimalkan F1 macro di VAL, lalu
  dipakai di test (metrik *_thr). Di data tidak seimbang, probabilitas yang terkalibrasi jarang melewati 0.5,
  sehingga F1 di ambang 0.5 merugikan model yang terkalibrasi. Ambang val diterapkan ke SEMUA metode.
Tambahan khusus sampel diffusion: 'agreement' = rata-rata proporsi sampel yang argmax-nya sama dengan
  kelas prediksi (ukuran ketidakpastian berbasis suara).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,
                             balanced_accuracy_score)

EPS = 1e-6
N_BINS_ECE = 15


# ----------------------------------------------------------------------
# Probabilitas dari sampel
# ----------------------------------------------------------------------
def probs_from_samples(Y, eps=EPS):
    """Y: sampel blok one-hot [N, n, K] (skala matriks, +-0.5). Kembali q [n, K] (jumlah 1)."""
    q = np.clip(Y.mean(0) + 0.5, eps, 1.0)
    return q / q.sum(1, keepdims=True)


def vote_agreement(Y, pred):
    """Proporsi sampel yang argmax-nya = kelas prediksi, per baris [n]."""
    return (Y.argmax(2) == pred[None, :]).mean(0)


def apply_temperature(q, T):
    z = np.log(np.clip(q, EPS, 1.0)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0)).mean())


def brier(P, y):
    return float(((P - np.eye(P.shape[1])[y]) ** 2).sum(1).mean())


def fit_temperature(q, y, lo=0.05, hi=20.0, objective='nll'):
    """T > 0 yang meminimalkan NLL (default) atau Brier ('brier') di data validasi (cari 1 dimensi pada log T).
    Catatan: untuk probabilitas kasar dari sedikit sampel (banyak 0/1), NLL didominasi baris yang salah
    dengan p ~ 0 sehingga T terdorong besar -> pakai objective='brier'."""
    f = nll if objective == 'nll' else brier
    res = minimize_scalar(lambda lt: f(apply_temperature(q, np.exp(lt)), y),
                          bounds=(np.log(lo), np.log(hi)), method='bounded')
    return float(np.exp(res.x))


# ----------------------------------------------------------------------
# Metrik
# ----------------------------------------------------------------------
def ece_score(P, y, n_bins=N_BINS_ECE):
    conf = P.max(1)
    hit = (P.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        sel = (conf > edges[b]) & (conf <= edges[b + 1]) if b else (conf >= 0) & (conf <= edges[1])
        if sel.any():
            ece += sel.mean() * abs(hit[sel].mean() - conf[sel].mean())
    return float(ece)


def aurc_score(P, y):
    """Area under risk-coverage curve: urutkan baris dari paling yakin, risiko = galat kumulatif."""
    order = np.argsort(-P.max(1), kind='stable')
    err = (P.argmax(1) != y)[order].astype(np.float64)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


# ----------------------------------------------------------------------
# Ambang keputusan (biner): dipilih di VAL untuk memaksimalkan F1, dipakai di test. Berlaku untuk SEMUA metode.
# ----------------------------------------------------------------------
def _f1_binary_all(p1, y, thrs):
    """F1 kelas 1 & kelas 0 untuk banyak ambang sekaligus (pred = p1 >= thr), eksak lewat pengurutan."""
    pp, pn = np.sort(p1[y == 1]), np.sort(p1[y == 0])
    n1, n0 = len(pp), len(pn)
    tp1 = n1 - np.searchsorted(pp, thrs, side='left')        # positif dengan p >= thr
    fp1 = n0 - np.searchsorted(pn, thrs, side='left')
    fn1 = n1 - tp1
    tp0, fp0, fn0 = n0 - fp1, fn1, fp1
    f1_1 = 2 * tp1 / np.maximum(2 * tp1 + fp1 + fn1, 1)
    f1_0 = 2 * tp0 / np.maximum(2 * tp0 + fp0 + fn0, 1)
    return f1_1, f1_0, n1, n0


def fit_threshold(P, y, objective='f1_macro'):
    """Biner saja (multikelas -> None, tetap argmax). Ambang di p(kelas 1) yang memaksimalkan F1 macro
    (atau 'f1_weighted') di data val. Kandidat = SEMUA nilai unik p di val (+0.5), dihitung eksak.
    Seri -> ambang terdekat ke 0.5."""
    if P.shape[1] != 2:
        return None
    p1 = P[:, 1]
    thrs = np.unique(np.concatenate([p1, [0.5]]))
    f1_1, f1_0, n1, n0 = _f1_binary_all(p1, y, thrs)
    score = (f1_1 + f1_0) / 2 if objective == 'f1_macro' else (f1_1 * n1 + f1_0 * n0) / (n1 + n0)
    best = np.flatnonzero(score >= score.max() - 1e-12)
    return float(thrs[best[np.argmin(np.abs(thrs[best] - 0.5))]])


def pred_at(P, thr=None):
    if thr is None or P.shape[1] != 2:
        return P.argmax(1)
    return (P[:, 1] >= thr).astype(np.int64)


def cls_metrics(P, y, thr=None):
    """P [n, K] probabilitas, y [n] indeks kelas benar. Metrik kelas memakai argmax (= ambang 0.5 untuk biner).
    Bila thr diberikan (biner), ditambah metrik kelas di ambang itu: acc_thr, f1_macro_thr, f1_weighted_thr,
    precision_w_thr, recall_w_thr (AUC/NLL/Brier/ECE tidak bergantung ambang)."""
    n, K = P.shape
    pred = P.argmax(1)
    onehot = np.eye(K)[y]
    out = {
        'acc': float((pred == y).mean()),
        'balanced_acc': float(balanced_accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0)),
        'f1_weighted': float(f1_score(y, pred, average='weighted', zero_division=0)),
        'precision_w': float(precision_score(y, pred, average='weighted', zero_division=0)),
        'recall_w': float(recall_score(y, pred, average='weighted', zero_division=0)),
        'nll': nll(P, y),
        'brier': float(((P - onehot) ** 2).sum(1).mean()),
        'ece': ece_score(P, y),
        'aurc': aurc_score(P, y),
        'entropy': float(-(P * np.log(np.clip(P, EPS, 1.0))).sum(1).mean()),
    }
    if K == 2:
        out['brier_biner'] = float(((P[:, 1] - y) ** 2).mean())
    for key, avg in (('auc', 'macro'), ('auc_weighted', 'weighted')):
        try:
            out[key] = float(roc_auc_score(y, P[:, 1]) if K == 2 else
                             roc_auc_score(y, P, multi_class='ovr', average=avg, labels=list(range(K))))
        except ValueError:                   # ada kelas yang tidak muncul di bagian ini
            out[key] = float('nan')
    if thr is not None and K == 2:
        pt = pred_at(P, thr)
        out.update({'thr': float(thr), 'acc_thr': float((pt == y).mean()),
                    'f1_macro_thr': float(f1_score(y, pt, average='macro', labels=[0, 1], zero_division=0)),
                    'f1_weighted_thr': float(f1_score(y, pt, average='weighted', zero_division=0)),
                    'precision_w_thr': float(precision_score(y, pt, average='weighted', zero_division=0)),
                    'recall_w_thr': float(recall_score(y, pt, average='weighted', zero_division=0))})
    return out


def fmt(m):
    return (f"Acc={m['acc']:.4f}  F1(macro)={m['f1_macro']:.4f}  F1(weighted)={m['f1_weighted']:.4f}  "
            f"Prec(w)={m['precision_w']:.4f}  Rec(w)={m['recall_w']:.4f}  AUC={m['auc']:.4f}  NLL={m['nll']:.4f}  "
            f"Brier={m['brier']:.4f}  ECE={m['ece']:.4f}  AURC={m['aurc']:.4f}"
            + (f"  | ambang {m['thr']:.3f}: Acc={m['acc_thr']:.4f}  F1(macro)={m['f1_macro_thr']:.4f}  "
               f"F1(weighted)={m['f1_weighted_thr']:.4f}" if 'thr' in m else ''))