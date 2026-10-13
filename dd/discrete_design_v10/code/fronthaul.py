"""Discrete design with a BS-specific stream set per BS (design round v10): separable core clusters under the fronthaul.

    design(net, p)          the proposed design (FS)
    design_1layer(net, p)   its 1-layer RSMA counterpart (FS1): the same selection without group-common streams

1. Core UEs: the order of semi-orthogonal user selection (SUS) on the stacked direct channels (thesis Sec. 4.1).
2. Core clusters. A core private stream that is carried by m BSs is precoded over Nm antennas and can be nulled at
   the other core UEs if Nm >= c, c the number of core streams; it occupies one fronthaul slot at each of the m BSs,
   and every BS keeps one slot for the super-common stream. The design serves the largest number c <= min(K, BN) of
   core UEs for which the fronthaul can give every core the same number of BSs,
        m = min(B, floor(B (S_max - 1) / c)),   with   N m >= c,
   each core UE (in SUS order) being carried by its m strongest BSs (direct-channel gain) that still have a free
   slot (fewer if fewer BSs have one).
3. Groups: every other UE joins the core UE with the most correlated direct channel (thesis Sec. 4.1).
4. Every BS carries the super-common stream and its core private streams, and fills its remaining slots with the
   group-common streams, then with the private streams of the non-core UEs, each in decreasing order of its
   direct-channel gain to the weakest decoder of the stream.
Nothing in it is tuned: c and m follow from N, B, S_max and K; the orders from the direct channels.
"""
import numpy as np
import hrs as H_


def _gain(Hd, p):
    """(K, B): ||f_{b,k}||^2, the direct-channel gain of BS b to UE k."""
    N = p['N']
    return np.stack([np.sum(np.abs(Hd[:, b * N:(b + 1) * N]) ** 2, axis=1) for b in range(p['B'])], axis=1)


def core_clusters(Hd, p):
    """Core UEs (SUS order), the BSs that carry each of them, and the cluster size m."""
    K, B, N, S = p['K'], p['B'], p['N'], p['Smax']
    g = _gain(Hd, p)
    order = H_.sus(Hd, p['BN'])[:min(K, p['BN'])]
    m_of = lambda c: min(B, B * (S - 1) // c)
    c = max(c for c in range(1, len(order) + 1) if N * m_of(c) >= c)
    m = m_of(c)
    load = np.zeros(B, int); carry = {}
    for k in order[:c]:
        carry[k] = [b for b in np.argsort(-g[k], kind='stable') if load[b] < S - 1][:m]
        load[carry[k]] += 1
    return order[:c], carry, m


def design(net, p, groups_on=True):
    Hd = net['Hd']; K, B, S = p['K'], p['B'], p['Smax']
    g = _gain(Hd, p)
    C, carry, m = core_clusters(Hd, p)
    groups = []
    if groups_on:
        e = np.sum(np.abs(Hd) ** 2, axis=1); Hn = Hd / np.sqrt(e)[:, None]
        follow = {a: [] for a in C}
        for k in range(K):
            if k not in C:
                follow[C[int(np.argmax(np.abs(Hn[k] @ Hn[C].conj().T)))]].append(k)
        groups = [[a] + follow[a] for a in C if follow[a] and len(follow[a]) + 1 < K]
    G = len(groups)
    dec = lambda j: groups[j - 1] if j <= G else [j - 1 - G]
    others = list(range(1, G + 1)) + [1 + G + k for k in range(K) if k not in C]   # group-common, non-core private
    sets = []
    for b in range(B):
        rank = sorted(others, key=lambda j: (j > G, -min(g[k, b] for k in dec(j)), j))
        sets.append(([0] + [1 + G + k for k in C if b in carry[k]] + rank)[:S])
    served = sorted(set().union(*map(set, sets)))
    d = H_.Design(groups, served, K)
    d.core = list(C); d.rule = 'fronthaul_clusters' + ('' if groups_on else '_1layer'); d.cluster = m
    d.priority = [0] + [1 + G + k for k in C] + others          # (stream clustering on this design takes this order)
    bm = np.array([[j in sets[b] for j in d.served] for b in range(B)])
    if not bm.all():
        d.bsmask = bm; d.bs_sets = [sorted(s) for s in sets]
    return d


def design_1layer(net, p):
    return design(net, p, groups_on=False)
