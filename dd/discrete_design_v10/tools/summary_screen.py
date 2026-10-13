"""Compact Q = 5 screen summary of the variants tried (design round v10): per scenario and R_th, the mean rate of the
variant, its paired difference to its 1-layer counterpart and to J (J derived from JH and J1 as in tools/report.py).
    python3 tools/summary_screen.py V1 C1 C3 GF D E ..."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'harness')); sys.path.insert(0, os.path.join(ROOT, 'tools'))
import variants as V, report as RP

for X in sys.argv[1:]:
    X1 = V.VARIANTS[X].get('one_layer')
    print(f'\n{X} (1-layer: {X1})')
    for sc in ('T2', 'E3'):
        d = {n: RP.load(n, 5).get(sc, {}) for n in (X, X1, 'JH', 'J1')}
        keys = set(d[X]) & set(d[X1]) & set(d['JH']) & set(d['J1'])
        row = []
        for R in RP_RTH if (RP_RTH := (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)) else ():
            ks = [k for k in keys if k[2] == R]
            a = [d[X][k]['rate'] - d[X1][k]['rate'] for k in ks]
            j = [d[X][k]['rate'] - RP.derive_j(d['JH'][k], d['J1'][k])['rate'] for k in ks]
            row.append(f'{np.mean([d[X][k]["rate"] for k in ks]):6.1f} ({np.mean(a):+5.2f}±{np.std(a, ddof=1) / np.sqrt(len(a)):4.2f}; J {np.mean(j):+6.2f})')
        print(f'  {sc} n={len(keys)}: ' + ' | '.join(row))
