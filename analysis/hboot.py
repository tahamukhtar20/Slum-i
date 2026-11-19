"""Hierarchical bootstrap of the paired change (Ours - baseline) over the 6 city-sensor datasets,
then the method x budget settings within each, then the 5 seeds within each setting.
Numbers in the Results section and the abstract. Usage: python analysis/hboot.py (under a minute)."""
import json, numpy as np
D = json.load(open('results/all.json')); S = [0, 42, 123, 999, 1337]
rng = np.random.default_rng(0)
def build(meths):
    X = {}
    for sat in ['esri', 's2']:
        for c in ['lahore', 'karachi', 'mumbai']:
            X[(sat, c)] = np.array([[[D[f'{sat}|{c}|{m}_freematchc|{b}|{s}'][i] - D[f'{sat}|{c}|{m}|{b}|{s}'][i]
                                      for s in S] for m in meths for b in ['10', '20', '30']] for i in (0, 1)])
    return X
def hboot(X, i, n=20000):
    keys = list(X); out = np.empty(n)
    for k in range(n):
        vals = []
        for kk in rng.choice(len(keys), len(keys)):
            A = X[keys[kk]][i]; st = rng.choice(A.shape[0], A.shape[0]); sd = rng.choice(5, (A.shape[0], 5))
            vals.append(A[st[:, None], sd].mean())
        out[k] = np.mean(vals)
    return out
for name, meths in [('all', ['fixmatch', 'unimatch', 'unimatchv2']), ('noFixMatch', ['unimatch', 'unimatchv2'])]:
    X = build(meths)
    for i, met in [(1, 'slum'), (0, 'mIoU')]:
        b = hboot(X, i); pt = np.mean([X[k][i].mean() for k in X]); lo, hi = np.percentile(b, [2.5, 97.5])
        p = 2 * min((b <= 0).mean(), (b >= 0).mean())
        print(f'{name:10s} {met:4s} mean {pt:+.2f} 95% CI [{lo:+.2f},{hi:+.2f}] p~{p:.4f}')
