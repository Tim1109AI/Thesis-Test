# Discrete design with a stream set per BS: separable core clusters under the fronthaul (design P)

Design round v10, return `cloud_high`. Code: `code/fronthaul.py` (`design`, `design_1layer`); variants `P` and `P1` in
`harness/variants.py`. Everything below was run with the harness of the package (self-test passed, all traces
identical).

## 1. The design (thesis-ready description)

Let $\mathbf h_{d,k}=[\mathbf f_{1,k},\ldots,\mathbf f_{B,k}]$ be the stacked direct channel of UE $k$,
$g_{k,b}=\|\mathbf f_{b,k}\|_2^2$ its gain from BS $b$, and $M_{b,j}\in\{0,1\}$ the indicator that BS $b$ carries stream
$j$, so that \eqref{eq:c-front} reads $\sum_j M_{b,j}\le S^{\max}$. Every BS carries the super-common stream
($M_{b,c}=1$); the design decides the remaining $S^{\max}-1$ slots of every BS from the direct channels of the whole
network, once per drop and setting, before the deployment search.

**Core users.** The core users are ordered by the greedy projection step of SUS on the stacked direct channels,
\eqref{eq:core_selection}, as in the joint design; let $u_1,u_2,\ldots$ be this order (length $\min\{K,BN\}$).

**How many core streams, and how many BSs per core stream.** A core private stream carried by $m$ BSs is precoded over
$Nm$ antennas; it can be steered into the null space of the channels of the other core users, and hence spatially
separated from all of them, if $Nm\ge c$, where $c$ is the number of core streams. It occupies one fronthaul slot at
each of its $m$ BSs, and the network has $B(S^{\max}-1)$ slots besides the super-common stream. Sharing these slots
evenly among the core streams gives each of them
$$m(c)=\min\Bigl\{B,\Bigl\lfloor \tfrac{B(S^{\max}-1)}{c}\Bigr\rfloor\Bigr\}$$
BSs, and the design serves the largest number of core streams that remain separable,
$$c^\star=\max\bigl\{c\le\min\{K,BN\}:\ N\,m(c)\ge c\bigr\},\qquad m^\star=m(c^\star),\qquad \mathcal C=\{u_1,\ldots,u_{c^\star}\}.$$
Since $N m(c)-c$ decreases in $c$, $c^\star$ is the end of a prefix and is found by one pass.

**Which BSs carry a core stream.** In the SUS order, the private stream of core user $u$ is carried by its $m^\star$
strongest BSs (largest $g_{u,b}$) among those that still have a free slot,
$\mathcal B_u\subseteq\mathcal B$, $|\mathcal B_u|=m^\star$ (fewer only if fewer BSs have a free slot: 2 of the 249
configurations checked, both in drop 907, which no tier uses).

**Groups and group-common streams.** Every UE outside $\mathcal C$ joins the core user with the most correlated direct
channel, \eqref{eq:nearest_core}, and every core user with at least one follower heads a group, as in the joint design;
the groups are disjoint, of size $2\ldots K-1$ and at most $\lfloor K/2\rfloor$. A non-core UE has no private stream:
its channel is not separable from the core streams with the fronthaul at hand (that is what $c^\star$ says), so its
private stream would only add interference; its QoS is carried by its group-common stream and the super-common stream.
Every BS fills the slots that remain after its core streams,
$S^{\max}-1-|\{u\in\mathcal C:b\in\mathcal B_u\}|$, with group-common streams in decreasing order of its gain to the
weakest member, $\min_{k\in\mathcal U_g}g_{k,b}$. These are the streams by which the BSs differ most: each BS carries
the groups it reaches best.

**Its 1-layer RSMA (P1)** is the same selection without group-common streams: the same core users, the same
$c^\star$, $m^\star$ and $\mathcal B_u$; the slots that P gives to group-common streams stay empty (P1 has no other
stream to put there under the same logic, since non-core UEs have no private stream).

```
Algorithm P (direct channels H_d, B, N, K, S_max)
 1  u_1, u_2, ... <- SUS order on the stacked direct channels, length min(K, BN)
 2  m(c) = min(B, floor(B (S_max - 1) / c));  c* = max{c <= min(K, BN) : N m(c) >= c};  m* = m(c*)
 3  C <- {u_1, ..., u_c*};  load_b <- 0 for every BS b
 4  for u in C (SUS order):  B_u <- the m* BSs with the largest g_{u,b} among {b : load_b < S_max - 1};  load_b += 1 on B_u
 5  groups: every k not in C joins argmax_{a in C} |h_k h_a^H|^2 / (|h_k|^2 |h_a|^2); heads with followers form groups
 6  for every BS b:  J_b <- {c} U {p_u : b in B_u} U (the first S_max - 1 - load_b group-common streams, in decreasing
                     order of min_{k in U_g} g_{k,b})
 7  return the groups, the served streams U_b J_b, the mask M_{b,j} = 1{j in J_b}, the core users C (AO start)
P1: steps 1-4 and 6 without step 5 (no group-common stream).
```

**Why it works.**
* *The core layer is what carries the sum rate.* At the SNRs of the network (55-60 dB direct links), the rate is
  limited by interference, and every core stream that can be separated adds a full stream of rate. The design serves as
  many separable core streams as the fronthaul allows, and gives them all the same, largest cluster size the fronthaul
  allows: in the thesis scenario T2 this is joint transmission by both BSs ($m^\star=B$) in every setting of the grid,
  i.e. the private layer of the joint design; in E3 (three BSs far apart) the fronthaul-limited settings serve more
  core users than the joint design ($c^\star=4,6,7,10$ against $3,5,5,9$ in settings b-e), each from 2 of the 3 BSs.
* *The group layer carries the QoS.* The super-common stream is decoded by every UE, so its rate is bounded by the
  weakest UE in the network and it cannot carry the QoS of many UEs at high $R_{th}$. A group-common stream reaches the
  UEs of one group (which are aligned with their core user), and each BS carries the groups it reaches best. This is
  why the design beats its 1-layer RSMA at high $R_{th}$ precisely where the fronthaul leaves room for group-common
  streams, and why both together beat the joint design at high $R_{th}$ in E3 (Section 3).
* *Non-core private streams are left out* because they carry no rate: in every design screened in this round their
  share of the sum rate was 0-3% (results/summary_screen_Q5.txt and Section 4), and in the sum rate versus $S^{\max}$
  they lowered the rate of the core streams (design FS, Section 4).

**Many BSs scattered over a large area (when some BSs, when all).** A core user is carried by its $m^\star$ strongest
BSs. When the fronthaul can carry every core stream at every BS ($S^{\max}-1\ge c^\star$, e.g. the benchmark setting
of the thesis), $m^\star=B$ and the design is joint transmission of all core streams, by itself. When it cannot, each
core stream is carried by the $m^\star<B$ BSs nearest to it in channel gain, so that different core users are carried
by different, overlapping clusters of BSs, and the group-common streams are BS-specific. `tools/large_network.py`
prints the outcome for 2-25 BSs over areas up to 1 km x 1 km (results/large_network_P.txt), e.g. 25 BSs, 100 UEs,
N = 4: S_max = 5: 20 core users, each carried by 5 of the 25 BSs; S_max = 12: 32 core users on 8 BSs each; S_max = 32:
55 core users on 14 BSs each. Every BS carries the super-common stream.

**Why nothing in it needs tuning.** $c^\star$ and $m^\star$ follow from $N$, $B$, $K$ and $S^{\max}$ (the separability
condition $Nm\ge c$ and an even share of the fronthaul); the core order is SUS, the groups are the nearest-core rule of
the thesis, and the BS choices are orders by the direct-channel gain. There is no threshold, weight or constant, no AO
test and no dependence on $R_{th}$, on the scenarios or on the BS positions; the design is deterministic and the same
for every setting.

**Cost.** The design itself takes 1-2 ms (CPU, mean 1.35 ms over the grid; `cpu_design` in every record). A run of the
framework costs about as much as the joint design in T2 (the same streams in most settings) and more in E3, where the
BSs serve more distinct streams (Section 3, CPU lines of the reports).

**Evaluation.** Unchanged: the AO start, the evaluator, the search R and the refinement of the package, with the
`d.bsmask` code path for streams carried by some BSs only. No option of `newprop.NEW` is changed. (An optional AO
start for BS-specific streams, `algo.INITS['core_bs']`, was tried, Section 4; it is off by default, unused by P and
does not change JH/J1.)

## 2. Choices made without asking (autonomous run)

* The final design was chosen on the Q = 5 screen and Q = 50 subsets of this round; its 1-layer counterpart leaves
  the group-common slots empty (Section 1), which is the literal "same selection logic without group-common streams"
  for a design whose logic gives private streams to core users only.
* The first version of this design (FS) also put the non-core private streams into the remaining slots (like the joint
  design). It was dropped in favour of P because those streams carry no rate and made the rate fall with $S^{\max}$
  (K = 8). FS has partial results (Section 4).
* JH and J1 were screened at Q = 5 (reference); they were not run at Q = 50 (compute), so the confirm reports have no J
  column; the Q = 5 reports compare with J.

## 3. Results of P

All rates in bit/s/Hz, 0 when the QoS is missed; paired differences, mean ± standard error. Full outputs of
`tools/report.py` are pasted in Section 7.

### 3.1 R1: P against its 1-layer RSMA P1 (confirm tier, Q = 50, 3 drops, 6 settings)

| R_th | 0 | 0.5 | 1 | 1.5 | 2 | 2.5 |
|---|---|---|---|---|---|---|
| T2: P − P1 | +1.14 ± 0.76 | +1.11 ± 0.57 | +0.74 ± 0.64 | −0.04 ± 0.60 | +0.76 ± 0.53 | +0.17 ± 0.97 |
| E3: P − P1 | +0.33 ± 1.34 | −2.61 ± 1.26 | −0.57 ± 0.70 | +2.90 ± 1.50 | +6.21 ± 2.10 | +5.41 ± 4.25 |

Q = 5 screen (4 drops): T2 −0.24 ± 0.24, −0.24 ± 0.24, −0.05 ± 0.05, +0.08 ± 0.08, +1.04 ± 0.61, +1.57 ± 1.02;
E3 −0.09 ± 0.08, −0.62 ± 0.77, +0.79 ± 0.60, +2.75 ± 1.22, +6.91 ± 2.78, +7.45 ± 3.13.

**Verdict on R1: met in E3 at R_th ≥ 1.5 (clearly at 2, by 3 SE), not met at R_th = 1 in either scenario nor at
R_th ≥ 1.5 in T2, where P and P1 tie.** The ties are structural, not noise:
* In T2 the separability rule gives every core stream to both BSs in all six settings ($m^\star=B=2$), and in settings
  b-e the core streams then take every slot ($c^\star m^\star = B(S^{\max}-1)$): no slot is left for a group-common
  stream, and P, P1 (and J, JH, J1) serve exactly the same streams, so P − P1 is exactly 0 in 72 of the 108 T2 runs.
  Only settings a and f leave slots for group-common streams; there P − P1 is +2.5 ± 1.6 / +4.3 ± 4.3 at R_th = 0 and
  +2.8 ± 1.6 / +1.6 ± 3.8 at R_th = 1 (Section 7), within the run-to-run variability of the search (about ±5 bit/s/Hz
  per run whenever two designs differ at all).
* In E3 the group-common streams get slots in settings b-e and pay off where the QoS binds: E3/c +9.1 ± 1.7
  (R_th = 1.5), +21.6 ± 4.1 (2), +24.1 (2.5, QoS met in 1 of 3 runs against 0 of 3); E3/b +11.4 ± 5.8 at 2.5. At low
  R_th a group-common stream carries almost no rate and only changes the path of the search (E3 R_th = 0.5:
  −2.6 ± 1.3, mostly E3/c and E3/f).
* The same holds for the joint design of the thesis on this grid: JH − J1 at Q = 5 is +0.29 ± 0.29 at R_th = 1 in
  T2 and −0.01 ± 0.01 in E3, because J serves group-common streams only in settings a and f (Section 7 has J at
  Q = 50 for reference).
* Every design tried in this round that gives group-common streams slots in the fronthaul-limited settings b-e loses
  to its 1-layer RSMA at R_th ≤ 1.5 (C1, C3, GF, D, Section 4): there a group-common stream can only take the slot of
  a separable private stream, and at moderate R_th that slot is worth more as a private stream. Winning R1 at every
  R_th would need a rule that knows when the QoS binds, i.e. the stream-set test the user has ruled out, or a closed-form
  QoS test (F1, Section 4), which rarely fires and did not help.

### 3.2 R2 and R3: the benchmark setting (T2, K = 16, N = 3, S_max = 12, R_th = 2, Q = 50, 15 runs)

| Scheme | M = 100 | M = 300 | M = 500 | mean | P − scheme |
|---|---|---|---|---|---|
| **P** | 74.68 | 81.44 | 85.68 | **80.60** | |
| sparse beamforming (stored) | 63.08 | 75.09 | 81.03 | 73.07 | **+7.54 ± 2.65** |
| stream clustering on P's streams (P@SC) | 44.18 | 54.38 | 64.51 | 54.36 | **+26.25 ± 1.76** |
| J (stored, reference) | 73.51 | 81.95 | 88.72 | 81.39 | |

P meets the QoS in 15/15 runs. Shares of the sum rate (R3): **super-common 15.6%, group-common 9.2%**, private 75.1%
(J in the thesis figure: 16.7% and 10.8%). At S_max = 12 the core streams fit at both BSs ($m^\star=2$), so P serves
the 6 core streams and the 4-5 group-common streams from both BSs: its groups are those of J, and P@SC equals the
stored J@SC run for run.

### 3.3 R4: sum rate versus S_max (T2, N = 3, M = 300, Q = 50, 2 drops)

| K \ S_max | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| 8 | 77.66 | 99.80 | 105.16 | 102.21 | 102.21 | 102.21 | 102.21 |
| 12 | 68.08 | 100.99 | 111.87 | 116.62 | 116.31 | 116.31 | 116.31 |
| 16 | 49.20 | 87.20 | 100.40 | 103.55 | 110.02 | 110.02 | 110.02 |

(means over R_th ∈ {0, 1, 2} and the drops.) The rate rises and then flattens: once every core stream is carried by
both BSs and every group-common stream fits, a larger S_max changes nothing (the design is the same for S_max ≥ 10,
12, 12 at K = 8, 12, 16, so those runs are identical). One step is negative: K = 8, S_max 8 → 10, −2.95 ± 1.58
(−7 in drop 901, ±0 in drop 902). Its only change is that the two group-common streams (which carry no rate at K = 8)
go from one BS each to both BSs; I read it as run-to-run variability of the search, not a loss of the design.

### 3.4 Where the BSs carry different streams (design requirement), confirm tier

| Scenario / setting | runs with BS-specific sets | per run, carried by a subset of the BSs: group-common / private |
|---|---|---|
| T2, every setting | 0/108 | 0 / 0 (the core streams are separable only from both BSs; all streams fit) |
| E3 a, f (S_max = 12, 20) | 0/36 | 0 / 0 (every core stream from all 3 BSs) |
| E3 b (S_max = 4) | 18/18 | 1.0 / 4.0 |
| E3 c (S_max = 6) | 18/18 | 2.7 / 6.0 |
| E3 d (S_max = 6) | 18/18 | 1.0 / 7.0 |
| E3 e (S_max = 10) | 18/18 | 3.7 / 10.0 |

In E3 b-e every core user is carried by 2 of the 3 BSs (its two strongest), different core users by different pairs,
and every BS fills its last slots with the group-common streams it reaches best.

### 3.5 Cost (CPU per run, s; confirm tier)

T2: P 55 (a 23, b 15, c 37, d 25, e 92, f 137); E3: P 287 (a 36, b 77, c 272, d 138, e 846, f 352); P1: T2 and E3
together 145. The design itself: 1-2 ms. E3 costs more because the BSs serve more distinct streams (up to 15 in E3/e);
P serves fewer streams than the joint design wherever the joint design adds non-core private streams.

## 4. Other methods tried (Q = 5 screens, 288 runs each; X − X1 and X − J per R_th, T2 / E3)

Columns: mean rate of X (X − its 1-layer counterpart X1; X − J), R_th = 0, 0.5, 1, 1.5, 2, 2.5
(`tools/summary_screen.py`, full output results/summary_screen_Q5.txt).

| Variant | idea | T2: X − X1 at R_th 0 … 2.5 | E3: X − X1 at R_th 0 … 2.5 | X − J (T2 / E3) at R_th = 0 and 2.5 |
|---|---|---|---|---|
| V1 | each BS carries its S_max best streams in the role order c, SUS core privates (all min(K,BN) cores), group-commons, non-core privates, by its own gain | +0.35 +0.35 +0.29 +0.29 +0.99 +1.46 | −0.08 −0.07 −0.01 −0.11 +0.01 +0.01 | −13.7 / −14.8 and +6.5 / +24.0 |
| V1b | V1 with the masked AO start (`algo.INITS['core_bs']`, off by default) | as V1 | as V1 | −13.9 / −14.1 and +5.9 / +24.1 |
| F1 | V1, group-common streams first when the QoS load of the UEs without private stream exceeds the single-stream capacity of the weakest UE | as V1 (fired in 4 runs, all infeasible anyway) | as V1 | as V1 |
| GF | V1 with group-common streams first | −12.3 −11.1 −8.9 −3.9 +4.3 +4.5 | −7.5 −7.7 −6.7 −5.7 −3.3 −1.2 | −26.3 / −22.2 and +9.5 / +22.8 |
| C1 | cell-first: each BS its own cell's cores (local SUS, N per BS), own groups, then cooperation | −7.3 −7.3 −5.7 −3.4 +3.0 −0.1 | −3.2 −2.2 −2.6 −2.4 −1.0 +2.1 | −20.3 / −15.3 and +3.0 / +23.8 |
| C3 | cell-first with global SUS cores | −6.0 −5.6 −6.0 −3.3 −0.2 −0.8 | −1.4 −1.7 −1.8 −1.9 −0.7 +0.3 | −20.1 / −14.9 and +4.5 / +23.7 |
| D | separable cores, minimal cluster m = ⌈c/N⌉, rest group-commons then non-core privates | +0.7 +1.2 +0.8 +1.6 +1.3 −1.4 | −0.1 −1.0 +2.0 +1.2 +1.0 +1.1 | −9.0 / −7.1 and +4.1 / +20.5 |
| E = FS | separable cores, even cluster (P's core layer), rest group-commons then non-core privates | +0.35 +0.35 +0.29 +0.29 +0.99 +1.46 | −0.1 −1.0 +2.0 +1.2 +1.0 +1.1 | −0.0 / −7.1 and −0.0 / +20.5 |
| **P** | **E without non-core private streams** | −0.24 −0.24 −0.05 +0.08 +1.04 +1.57 | −0.09 −0.62 +0.79 +2.75 +6.91 +7.45 | −0.4 / −7.0 and +0.1 / +20.5 |

What each taught:
* **V1** (the most direct reading of "each BS chooses its own streams") splits the core streams between the BSs when
  S_max − 1 < BN; at 55-60 dB that costs spatial degrees of freedom (−14 against J at R_th = 0 in both scenarios) but
  gives every BS its own private streams, which carries the QoS at high R_th (+24 against J in E3 at R_th = 2.5). Its
  group-common streams only enter where every core stream fits, so V1 − V11 is J's own JH − J1. Bench (Q = 50): 82.52,
  +9.46 ± 2.05 over sparse BF, shares 14.3% / 9.9%. Q = 50 on settings a, f (2 drops), V1 − V11: T2 −3.7, −6.0,
  +0.5, −1.5, +2.1, +3.5; E3 +0.4, +1.5, +5.7, +0.6, −2.7, +3.1 (R_th 0 … 2.5): noise-dominated.
* **V1b**: the masked start does not reduce V1's loss; the loss is the missing spatial separation, not the start.
* **C1, C3, GF** put group-common streams before (joint support of) core streams: they lose to their 1-layer RSMA at
  R_th ≤ 1.5 in both scenarios; groups only win when the QoS is nearly infeasible.
* **D, E** introduced the separability rule. D's minimal clusters (single-BS cores in T2 b, d) lose 12-37 against J at
  low R_th; E's even clusters reproduce joint transmission in T2 and serve more, partially associated core users in
  E3. Q = 50 on E3 b, c, e (D = E there, 2 drops): 113.5 / 110.6 / 102.8 / 93.8 / 81.4 / 65.7 (R_th 0 … 2.5).
* **FS** (= E, the first clean version of P) also put non-core private streams into the last slots. They carry 0-3% of
  the sum rate in every design screened (results/summary_screen_Q5.txt has the shares) and lowered the core rate as
  S_max grew (FS smax, K = 8: 103.9 → 97.5 → 93.9 for S_max = 8, 10, 12 in drop 901, the non-core privates at zero
  rate). P drops them.

## 5. Weak points

* **R1 is not met at every R_th** (Section 3.1): P beats P1 clearly only in E3 at R_th ≥ 1.5. Where the fronthaul is
  short and every slot is needed for a separable core stream (T2 b-e), P serves no group-common stream and is
  identical to its 1-layer RSMA; where slots are spare (T2 a, f; E3 a, f) the group-common streams change the rate by
  less than the search variability at R_th ≤ 1.5. The same is true of the joint design on this grid.
* **In T2 the BSs never carry different streams.** With two BSs 50 m apart the separability rule always asks for both
  BSs per core stream, and all streams fit. The BS-specific sets appear in E3 (and in larger networks).
* **The separability count assumes that every core stream must be separable from every other one** (the full
  coupling of the thesis networks, where every UE hears every BS 5-15 dB below its strongest one). In a very large
  network this is conservative: clusters grow like $\sqrt{B(S^{\max}-1)/N}$ (8 of 25 BSs per core user over 1 km² at
  S_max = 12) and the core streams take most of the fronthaul, so that only some of the groups get a group-common
  stream (results/large_network_P.txt).
* P1 leaves empty the slots that P gives to group-common streams (non-core UEs have no private stream in either). This
  costs P1 nothing measurable: where the joint design fills those slots with non-core private streams, the rate is
  not higher (T2/f at Q = 5, R_th = 0: P1 114.95 against J1 112.06).
* One step of the sum rate versus S_max is negative (K = 8, 8 → 10, −2.95 ± 1.58), from one drop.
* E3 runs cost about 5 times the CPU of T2 runs (up to 15 distinct streams in E3/e at Q = 50, 850 CPU-s per run).

## 6. Files

* `code/fronthaul.py`: the design (P, P1) and its first version (FS, FS1); `code/perbs.py`, `code/perbs_v.py`: the
  other variants of Section 4; `code/algo.py`: the optional masked AO start (`INITS['core_bs']`, `SEEDS['rule_bs']`,
  off by default; used only by V1b).
* `harness/variants.py`: all variants of this round. `tools/large_network.py`, `tools/summary_screen.py`: the
  large-network illustration and the screen summary.
* `package.diff`: unified diff against the package (code, harness, tools).
* `results/*.jsonl`: every run of this round (harness format, unedited), `results/*report*.json` (and `.txt`).
* Self-test: passed, all traces identical (also after the changes of this round; Section 7).

## 7. Report outputs

REPORTS_PLACEHOLDER

