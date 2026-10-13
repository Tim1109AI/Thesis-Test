"""How the design FS (code/fronthaul.py) associates streams with BSs in a large network of scattered BSs (no
simulation: the discrete design only, from the direct channels of the thesis channel model).

    python3 tools/large_network.py

Networks: B BSs at jittered grid positions over an L x L area (height 15 m), K UEs uniform (height 1.5 m), N antennas,
the channel constants of the thesis (code/hrs.py). For each S_max: the number c of core UEs, the cluster size m (BSs
per core UE), the number of BSs carrying each core (min / mean / max), how many BSs carry a core of their own (a core
carried by that BS only) or share it, the group-common and non-core private streams and how many distinct BSs carry
them, and the distinct streams per BS.
"""
import os, sys, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'code'))
import hrs as H_, run as R, fronthaul as F, discrete as DC


def network(B, K, N, L, seed):
    rng = np.random.default_rng(seed)
    side = int(math.ceil(math.sqrt(B)))
    cells = [((i + 0.5) * L / side, (j + 0.5) * L / side) for i in range(side) for j in range(side)][:B]
    BS = np.array([(x + rng.uniform(-0.2, 0.2) * L / side, y + rng.uniform(-0.2, 0.2) * L / side, 15.0) for x, y in cells])
    UE = np.column_stack([rng.uniform(0, L, K), rng.uniform(0, L, K), np.full(K, 1.5)])
    p = H_.params(K=K, N=N, M=100, Smax=4, B=B, bank=R.BANK)
    k0 = 2 * math.pi / p['lam']
    Hd = np.zeros((K, B * N), complex)
    for k in range(K):
        for b in range(B):
            v = UE[k] - BS[b]; d = np.linalg.norm(v)
            a = np.exp(1j * k0 * p['uBS'] @ (v / d))
            Hd[k, b * N:(b + 1) * N] = math.sqrt(p['PL0'] / p['noise']) * d ** (-p['alpha'] / 2) * np.exp(-1j * k0 * d) * a
    return dict(BS=BS, UE=UE, Hd=Hd), p


def describe(net, p):
    d = F.design(net, p); DC.check(d, p)
    B, G, K = p['B'], d.G, p['K']
    bm = d.bsmask if d.bsmask is not None else np.ones((B, d.nS), bool)
    nb = {j: int(bm[:, d.col[j]].sum()) for j in d.served}
    cores = [1 + G + k for k in d.core if (1 + G + k) in d.col]
    own = sum(1 for j in cores if nb[j] == 1)
    grp = [j for j in d.served if 1 <= j <= G]; ncp = [j for j in d.served if j > G and (j - 1 - G) not in d.core]
    cb = [nb[j] for j in cores]
    return (f"S_max={p['Smax']:2d}: c={len(d.core):3d} core UEs, m={d.cluster:2d}; BSs per core min/mean/max "
            f"{min(cb)}/{np.mean(cb):.1f}/{max(cb)} ({own} cores carried by one BS only, {len(cores) - own} by several); "
            f"group-common {len(grp)} of {G} (BSs each {np.mean([nb[j] for j in grp]) if grp else 0:.1f}), "
            f"non-core private {len(ncp)}; distinct streams {d.nS}, per BS {int(bm.sum(1).min())}..{int(bm.sum(1).max())}")


if __name__ == '__main__':
    for B, K, N, L in [(2, 16, 3, 100), (3, 16, 3, 100), (9, 36, 4, 300), (16, 64, 4, 600), (25, 100, 4, 1000)]:
        net, p = network(B, K, N, L, seed=B * 1000 + K)
        print(f'\n== B = {B} BSs over {L} m x {L} m, K = {K} UEs, N = {N}')
        for S in sorted({N + 1, 2 * N, 3 * N, 4 * N, 8 * N}):
            p['Smax'] = S
            print('  ' + describe(net, p))
