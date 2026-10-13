"""Exploratory BS-specific discrete designs (design round v10). See DESIGN.md for the final one."""
import numpy as np
import hrs as H_


def _gains(Hd, p):
    B, N = p['B'], p['N']
    return np.stack([np.sum(np.abs(Hd[:, b * N:(b + 1) * N]) ** 2, axis=1) for b in range(B)], axis=1)   # (K, B)


def _finish(groups, sets, K, B, core, pr, rule):
    served = sorted(set().union(*map(set, sets)))
    d = H_.Design(groups, served, K)
    d.core = list(core); d.priority = list(pr); d.rule = rule
    bm = np.array([[j in sets[b] for j in d.served] for b in range(B)])
    if not bm.all():
        d.bsmask = bm; d.bs_sets = [sorted(s) for s in sets]
    return d


def perbs_rank(net, p, groups_on=True):
    """V1: SUS cores and nearest-core groups (as the current design); every BS fills its S_max slots in the class order
    super-common, core privates, group-common, non-core privates, each class sorted by the BS's direct-channel gain to
    the weakest decoder of the stream."""
    Hd = net['Hd']; K, B, S = p['K'], p['B'], p['Smax']
    order, groups, energy, _ = H_.joint_structure(Hd, p, 'sus')
    if not groups_on: groups = []
    G = len(groups); g = _gains(Hd, p)
    dec = lambda j: list(range(K)) if j == 0 else (groups[j - 1] if j <= G else [j - 1 - G])
    a = lambda j, b: min(g[k, b] for k in dec(j))
    core_p = [1 + G + k for k in order]
    grp = [1 + i for i in range(G)]
    oth = [1 + G + k for k in range(K) if k not in order]
    sets = []
    for b in range(B):
        lst = [0]
        for cls in (core_p, grp, oth):
            lst += sorted(cls, key=lambda j: (-a(j, b), j))
        sets.append(lst[:S])
    pr = H_.priority_list(order, groups, energy, K, 'private_first') if groups_on else [0] + core_p + sorted(oth, key=lambda j: -energy[j - 1 - G])
    return _finish(groups, sets, K, B, order, pr, 'perbs_rank')


def perbs_rank_1l(net, p):
    return perbs_rank(net, p, groups_on=False)


# --------------------------------------------------------------------------- cell-first family (exploration)
def _anchor(dec, g):
    """BS with the largest direct-channel gain to the weakest decoder of a stream."""
    return int(np.argmax(np.min(g[dec], axis=0)))


def cell_first(net, p, groups_on=True, cores='local', order='O1', grouping='global'):
    """Every UE's anchor is its strongest BS (its cell). Core UEs: 'local' = SUS on each BS's own channel among its
    cell (at most N per BS), 'global' = SUS on the stacked channel (min(K, BN)). Groups: every non-core UE joins the core
    with the most correlated channel ('global': stacked channel; 'cell': the cores of its own cell, own-BS channel).
    Every stream is anchored at the BS with the largest gain to its weakest decoder. BS b carries, in order (first
    S_max): O1 = c, own cores, own groups, other cores, other groups, non-core privates;
         O2 = c, own cores, other cores, own groups, other groups, non-core privates;
         O3 = c, own cores, own groups, own non-core privates, other cores, other groups, other privates.
    'other' streams by the gain of b to their weakest decoder."""
    Hd = net['Hd']; K, B, S, N = p['K'], p['B'], p['Smax'], p['N']
    g = _gains(Hd, p); cell = np.argmax(g, axis=1)
    if cores == 'global':
        C = H_.sus(Hd, p['BN'])[:min(K, p['BN'])]
    else:
        C = []
        for b in range(B):
            U = [k for k in range(K) if cell[k] == b]
            if U:
                sel = H_.sus(Hd[U][:, b * N:(b + 1) * N], N)
                C += [U[i] for i in sel[:min(N, len(U))]]
    energy = np.sum(np.abs(Hd) ** 2, axis=1)
    Hn = Hd / np.sqrt(energy)[:, None]
    follow = {c: [] for c in C}
    for k in range(K):
        if k in C: continue
        if grouping == 'cell':
            own = [c for c in C if cell[c] == cell[k]] or C
            b = cell[k]; f = Hd[:, b * N:(b + 1) * N]; fn = f / np.linalg.norm(f, axis=1)[:, None]
            corr = [abs(fn[k] @ fn[c].conj()) for c in own]
            follow[own[int(np.argmax(corr))]].append(k)
        else:
            corr = np.abs(Hn[k] @ Hn[C].conj().T)
            follow[C[int(np.argmax(corr))]].append(k)
    groups = [[c] + follow[c] for c in C if follow[c]] if groups_on else []
    G = len(groups)
    dec = lambda j: list(range(K)) if j == 0 else (groups[j - 1] if j <= G else [j - 1 - G])
    a = lambda j, b: min(g[k, b] for k in dec(j))
    anc = {j: _anchor(dec(j), g) for j in range(1, 1 + G + K)}
    core_p = [1 + G + k for k in C]                          # in selection order
    grp = sorted(range(1, 1 + G), key=lambda j: (-len(dec(j)), j))
    oth = [1 + G + k for k in range(K) if k not in C]
    sets = []
    for b in range(B):
        own = lambda L: [j for j in L if anc[j] == b]
        other = lambda L: sorted([j for j in L if anc[j] != b], key=lambda j: (-a(j, b), j))
        bya = lambda L: sorted(L, key=lambda j: (-a(j, b), j))
        if order == 'O1':
            lst = [0] + own(core_p) + own(grp) + other(core_p) + other(grp) + bya(oth)
        elif order == 'O2':
            lst = [0] + own(core_p) + other(core_p) + own(grp) + other(grp) + bya(oth)
        elif order == 'O3':
            lst = [0] + own(core_p) + own(grp) + bya(own(oth)) + other(core_p) + other(grp) + other(oth)
        sets.append(lst[:S])
    pr = [0] + core_p + grp + sorted(oth, key=lambda j: -energy[j - 1 - G])
    return _finish(groups, sets, K, B, C, pr, f'cell_first_{cores}_{order}_{grouping}')


# --------------------------------------------------------------------------- QoS-feasibility switch (exploration)
def common_capacity(Hd, p):
    """Largest rate a single stream decodable by every UE can have (all BSs at full power, co-phased MRT per BS):
    log2(1 + P_max min_k (sum_b ||f_{b,k}||)^2)."""
    B, N = p['B'], p['N']
    amp = np.stack([np.linalg.norm(Hd[:, b * N:(b + 1) * N], axis=1) for b in range(B)], axis=1)
    return float(np.log2(1 + p['Pmax'] * np.min(amp.sum(1)) ** 2))


def perbs_feas(net, p):
    """V1, but every BS puts the group-common streams first (c, groups, cores, non-core privates) when the 1-layer
    set of V1 cannot meet the QoS: the UEs without a private stream need (count) x R_th from the super-common stream,
    more than common_capacity."""
    d1 = perbs_rank(net, p, groups_on=False)
    U = p['K'] - len(d1.private_served)
    if U * p['Rth'] <= common_capacity(net['Hd'], p):
        d = perbs_rank(net, p); d.rule = 'perbs_feas(private_first)'; return d
    Hd = net['Hd']; K, B, S = p['K'], p['B'], p['Smax']
    order, groups, energy, _ = H_.joint_structure(Hd, p, 'sus')
    G = len(groups); g = _gains(Hd, p)
    dec = lambda j: list(range(K)) if j == 0 else (groups[j - 1] if j <= G else [j - 1 - G])
    a = lambda j, b: min(g[k, b] for k in dec(j))
    core_p = [1 + G + k for k in order]; grp = [1 + i for i in range(G)]
    oth = [1 + G + k for k in range(K) if k not in order]
    sets = []
    for b in range(B):
        lst = [0]
        for cls in (grp, core_p, oth):
            lst += sorted(cls, key=lambda j: (-a(j, b), j))
        sets.append(lst[:S])
    pr = H_.priority_list(order, groups, energy, K, 'group_first')
    return _finish(groups, sets, K, B, order, pr, 'perbs_feas(group_first)')
