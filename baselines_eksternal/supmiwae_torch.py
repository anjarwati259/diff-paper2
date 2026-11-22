"""
supMIWAE (Ipsen, Mattei & Frellsen, ICLR 2022: "How to deal with missing data in supervised deep learning?")
PORT PyTorch dari kode resmi TensorFlow: https://github.com/nbip/supMIWAE
    models/mlp_miwae.py                         -> MLPMIWAE (encoder, decoder, discriminator)
    models/discriminators.py                    -> MLPClassifier
    estimators/variational_inference.py         -> ELBO IWAE (pretraining)
    estimators/supervised_variational_inference -> ELBO supervised + prediksi berbobot SNIS
    experiments/sklearn/task01.py + configs     -> prosedur & hyperparameter

Prosedur resmi (task01.py):
  0. Data distandardisasi, sel hilang diisi 0 (SimpleImputer constant 0), s = 1 untuk sel teramati.
  1. PRETRAIN MIWAE (encoder + decoder): maks ELBO IWAE, Adam 1e-3, batch 100, n_samples 50,
     simpan bobot terbaik menurut iwae_elbo val (dicek tiap eval_every langkah).
  2. TRAIN supMIWAE: generator DIBEKUKAN, hanya discriminator p(y|x) yang dilatih dengan ELBO supervised
        log w_k = log p(y|x~_k) + log p(x_obs|z_k) + log p(z_k) - log q(z_k|x),  ELBO = logmeanexp_k(log w_k)
     Adam 1e-4, batch 100, n_samples 50, simpan bobot terbaik menurut akurasi val (regresi: -MSE val).
  3. PREDIKSI: bobot SNIS hanya dari x (tanpa y): snis_k = softmax_k(log p(x_obs|z_k) + log p(z_k) - log q(z_k|x));
        klasifikasi  p(y|x) = sum_k snis_k softmax(logits_k);   regresi  y^ = sum_k snis_k loc_k.
     Imputasi tunggal = sum_k snis_k x~_k  (x~ = x_obs di sel teramati, sampel p(x|z) di sel hilang).

Arsitektur resmi: encoder/decoder/discriminator MLP [50, 50, 50] aktivasi ELU; Normal dengan
scale = softplus(.) + 1e-6; encoder menerima x (sudah diisi 0) saja.

Perbedaan dari kode resmi (dicatat apa adanya):
  - Bahasa PyTorch (kode resmi TF2 + tensorflow_probability).
  - Kode resmi memakai epoch tetap (10000 / 15000) untuk data mainan 5000 baris; di sini ada batas epoch
    + berhenti dini bila skor val tidak membaik `patience_eval` kali evaluasi berturut-turut.
  - n_latent default = d - 1 (aturan MIWAE, Mattei & Frellsen 2019, untuk data UCI); konfig mainan memakai 2.
  - Regresi: discriminator = Normal(loc, softplus(scale)) (kelas VariationalInferenceRegression resmi).
  - Prediksi memakai n_samples_test (default 1000; resmi 10000) per baris, diproses per batch.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def mlp(n_in, n_hidden, n_out):
    layers, i = [], n_in
    for h in n_hidden:
        layers += [nn.Linear(i, h), nn.ELU()]
        i = h
    layers.append(nn.Linear(i, n_out))
    return nn.Sequential(*layers)


def normal_logp(x, loc, scale):
    return -0.5 * ((x - loc) / scale) ** 2 - torch.log(scale) - 0.5 * LOG2PI


class NormalBlock(nn.Module):
    """BasicBlock resmi: MLP -> (loc, softplus(log_scale) + 1e-6)."""
    def __init__(self, n_in, n_out, n_hidden):
        super().__init__()
        self.n_out = n_out
        self.net = mlp(n_in, n_hidden, 2 * n_out)

    def forward(self, x):
        o = self.net(x)
        return o[..., :self.n_out], F.softplus(o[..., self.n_out:]) + 1e-6


class SupMIWAE(nn.Module):
    def __init__(self, d, task, n_classes=2, n_latent=None, n_hidden=(50, 50, 50)):
        super().__init__()
        self.task, self.d = task, d
        self.n_latent = n_latent or max(1, d - 1)
        self.encoder = NormalBlock(d, self.n_latent, list(n_hidden))
        self.decoder = NormalBlock(self.n_latent, d, list(reversed(n_hidden)))
        if task == 'classification':
            self.disc = mlp(d, list(n_hidden), n_classes)          # MLPClassifier resmi (logits)
        else:
            self.disc = NormalBlock(d, 1, list(n_hidden))         # Normal(loc, scale) untuk y

    # ---- MLPMIWAE.call + bagian log-prob estimator ----
    def forward(self, x, s, n_samples):
        loc_z, sc_z = self.encoder(x)
        eps = torch.randn(n_samples, *loc_z.shape, device=x.device)
        z = loc_z + sc_z * eps                                          # [S, B, z]
        loc_x, sc_x = self.decoder(z)                                   # [S, B, d]
        x_samples = loc_x + sc_x * torch.randn_like(loc_x)
        x_mixed = s * x + (1 - s) * x_samples
        lpz = normal_logp(z, torch.zeros_like(z), torch.ones_like(z)).sum(-1)
        lqzx = normal_logp(z, loc_z, sc_z).sum(-1)
        lpxz = (normal_logp(x, loc_x, sc_x) * s).sum(-1)
        return {'x_mixed': x_mixed, 'iwae_log_w': lpxz + lpz - lqzx}

    def log_pyx(self, x_mixed, y):
        if self.task == 'classification':
            lp = torch.log_softmax(self.disc(x_mixed), -1)              # [S, B, K]
            return lp.gather(-1, y.view(1, -1, 1).expand(lp.shape[0], -1, 1)).squeeze(-1)
        loc, sc = self.disc(x_mixed)
        return normal_logp(y.view(1, -1), loc[..., 0], sc[..., 0])

    @staticmethod
    def logmeanexp(a, dim=0):
        return torch.logsumexp(a, dim) - math.log(a.shape[dim])

    def iwae_loss(self, x, s, n_samples):
        o = self(x, s, n_samples)
        return -self.logmeanexp(o['iwae_log_w']).mean()

    def sup_loss(self, x, s, y, n_samples):
        o = self(x, s, n_samples)
        return -self.logmeanexp(self.log_pyx(o['x_mixed'], y) + o['iwae_log_w']).mean()

    @torch.no_grad()
    def predict(self, x, s, n_samples, n_draw=0):
        """-> (prediksi, imputasi SNIS, sampel y [n_draw, B] untuk regresi atau None)."""
        o = self(x, s, n_samples)
        snis = torch.softmax(o['iwae_log_w'], 0)                         # [S, B]
        imp = (snis.unsqueeze(-1) * o['x_mixed']).sum(0)
        if self.task == 'classification':
            prob = (snis.unsqueeze(-1) * torch.softmax(self.disc(o['x_mixed']), -1)).sum(0)
            return prob, imp, None
        loc, sc = self.disc(o['x_mixed'])
        loc, sc = loc[..., 0], sc[..., 0]
        pred = (snis * loc).sum(0)
        draws = None
        if n_draw > 0:                                                    # sampel dari campuran berbobot SNIS
            k = torch.multinomial(snis.T, n_draw, replacement=True).T     # [n_draw, B]
            draws = loc.gather(0, k) + sc.gather(0, k) * torch.randn(k.shape, device=x.device)
        return pred, imp, draws


def _batches(n, bs, device, shuffle=True):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def _train_stage(model, params, lr, loss_fn, score_fn, n, cfg, device, log, name):
    """Loop trainer.py resmi: evaluasi tiap eval_every langkah, simpan bobot dengan skor val TERBESAR."""
    opt = torch.optim.Adam(params, lr=lr)
    best, best_state, bad, step = -float('inf'), None, 0, 0
    for ep in range(cfg['max_epochs']):
        model.train()
        for idx in _batches(n, cfg['batch_size'], device):
            loss = loss_fn(idx)
            opt.zero_grad(); loss.backward(); opt.step()
            if step % cfg['eval_every'] == 0:
                model.eval()
                sc = float(score_fn())
                model.train()
                if sc > best:
                    best, bad = sc, 0
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
            step += 1
            if bad >= cfg['patience_eval']:
                break
        if bad >= cfg['patience_eval']:
            break
    model.load_state_dict(best_state)
    model.eval()
    log[name] = {'skor_val_terbaik': best, 'langkah': step, 'epoch': ep + 1}


def fit_supmiwae(task, Xtr, Str, ytr, Xva, Sva, yva, n_classes, cfg, device):
    """X: standardisasi, sel hilang = 0; S: 1 = teramati. y: indeks kelas (long) / float (skala model)."""
    T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
    ydt = torch.long if task == 'classification' else torch.float32
    Xtr, Str, ytr = T(Xtr), T(Str), T(ytr, ydt)
    Xva, Sva, yva = T(Xva), T(Sva), T(yva, ydt)
    d = Xtr.shape[1]
    model = SupMIWAE(d, task, n_classes, cfg.get('n_latent'), cfg['n_hidden']).to(device)
    log, S = {}, cfg['n_samples']

    def val_eval(fn):
        with torch.no_grad():
            return fn()

    # 1) pretrain MIWAE (encoder + decoder), skor = iwae_elbo val
    gen = list(model.encoder.parameters()) + list(model.decoder.parameters())
    _train_stage(model, gen, cfg['lr_pretrain'],
                 lambda i: model.iwae_loss(Xtr[i], Str[i], S),
                 lambda: val_eval(lambda: -model.iwae_loss(Xva, Sva, S)),
                 len(Xtr), cfg, device, log, 'pretrain_miwae')

    # 2) supMIWAE: generator beku, hanya discriminator; skor = akurasi val (regresi: -MSE val)
    for p in gen:
        p.requires_grad_(False)

    def score():
        out, _, _ = predict_batched(model, Xva, Sva, S, 0, cfg['batch_pred'])
        if task == 'classification':
            return (out.argmax(1) == yva).float().mean()
        return -((out - yva) ** 2).mean()
    _train_stage(model, list(model.disc.parameters()), cfg['lr_sup'],
                 lambda i: model.sup_loss(Xtr[i], Str[i], ytr[i], S),
                 score, len(Xtr), cfg, device, log, 'supmiwae')
    log.update({'n_latent': model.n_latent, 'n_hidden': list(cfg['n_hidden'])})
    return model, log


def predict_batched(model, X, S, n_samples, n_draw, bs):
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=next(model.parameters()).device)
    X, S = T(X), T(S)
    outs, imps, draws = [], [], []
    for i in range(0, len(X), bs):
        o, imp, dr = model.predict(X[i:i + bs], S[i:i + bs], n_samples, n_draw)
        outs.append(o); imps.append(imp)
        if dr is not None:
            draws.append(dr)
    return torch.cat(outs), torch.cat(imps), (torch.cat(draws, 1) if draws else None)
