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
its channel is not separable from the core streams with the antennas and the fronthaul at hand (that is what
$c^\star$ says), so its private stream would only add interference; its QoS is carried by its group-common stream and the super-common stream.
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
* JH and J1 were screened at Q = 5 and, at the end, run at Q = 50 on the confirm tier, so that the confirm report
  has the J column (Section 7).

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
* The same holds for the joint design of the thesis on this grid (JH and J1 run at Q = 50 in this round): J − J1 is
  +0.92 ± 0.47, +0.55 ± 0.64, +0.59 ± 0.46, +0.02 ± 0.27, +0.32 ± 0.18, +1.25 ± 0.89 in T2 and +0.41 ± 0.41,
  +0.70 ± 0.70, +0.60 ± 0.60, +0.79 ± 0.55, −0.15 ± 0.73, +0.27 ± 0.27 in E3 (R_th = 0 … 2.5), because J serves
  group-common streams only in settings a and f. P's gain over its 1-layer RSMA at high R_th in E3 (+6.2, +5.4) is
  several times J's.
* P against J (same runs, Q = 50): T2 −0.39 ± 0.44, −0.54 ± 0.69, −0.24 ± 0.63, +0.12 ± 0.89, −0.31 ± 0.57,
  −0.26 ± 0.47 (in T2 b-e P serves exactly J's streams; in a and f, J's streams without the non-core private streams, and
  without J's stream-set test); E3 −9.31 ± 5.41,
  −5.07 ± 5.23, −0.51 ± 5.19, +5.86 ± 4.30, +18.21 ± 5.57, +20.81 ± 5.99, QoS met 106/108 against 100/108.
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
| T2, every setting | 0/108 | 0 / 0 (every core stream from both BSs; the group-common streams all fit (a, f) or none does (b-e)) |
| E3 a, f (S_max = 12, 20) | 0/36 | 0 / 0 (every core stream from all 3 BSs) |
| E3 b (S_max = 4) | 18/18 | 1.0 / 4.0 |
| E3 c (S_max = 6) | 18/18 | 2.7 / 6.0 |
| E3 d (S_max = 6) | 18/18 | 1.0 / 7.0 |
| E3 e (S_max = 10) | 18/18 | 3.7 / 10.0 |

In E3 b-e every core user is carried by 2 of the 3 BSs (its two strongest), different core users by different pairs,
and every BS fills its last slots with the group-common streams it reaches best.

### 3.5 Cost (CPU per run, s; confirm tier)

T2: P 55 (a 23, b 15, c 37, d 25, e 92, f 137); E3: P 287 (a 36, b 77, c 272, d 138, e 846, f 352); P1: T2 48, E3 145; J: T2
68, E3 98. The design itself: 1-2 ms. E3 costs more because the BSs serve more distinct streams (up to 15 in E3/e);
P serves fewer streams than the joint design wherever the joint design adds non-core private streams.

## 4. Other methods tried (Q = 5 screens, 288 runs each; X − X1 and X − J per R_th, T2 / E3)

Paired differences of X against its 1-layer counterpart X1 per R_th (0, 0.5, 1, 1.5, 2, 2.5) and against J at R_th = 0
and 2.5 (`tools/summary_screen.py`; the full output, with the mean rates and all R_th, is results/summary_screen_Q5.txt).

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
* **D, E** introduced the separability rule. D's minimal clusters (single-BS cores in T2 b, d) lose 17-37 against J at
  R_th = 0; E's even clusters reproduce joint transmission in T2 and serve more, partially associated core users in
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

### R1, confirm tier (Q = 50), with --by-setting

`python3 tools/report.py P --by-setting --json`

```

===== scenario T2, Q = 50: 108 common runs (P: 108, P1: 108, JH: 108, J1: 108)
-- T2, all settings
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0   18    124.29    123.15    124.68    123.91    123.76    +1.14± 0.76       -0.39± 0.44      18/18/18/18/18
    0.5   18    120.87    119.76    121.41    120.44    120.85    +1.11± 0.57       -0.54± 0.69      18/18/18/18/18
      1   18    113.30    112.56    113.54    113.02    112.95    +0.74± 0.64       -0.24± 0.63      18/18/18/18/18
    1.5   18    103.73    103.78    103.61    103.47    103.59    -0.04± 0.60       +0.12± 0.89      18/18/18/18/18
      2   18     93.23     92.47     93.54     94.06     93.22    +0.76± 0.53       -0.31± 0.57      18/18/18/18/18
    2.5   18     76.60     76.42     76.86     77.34     75.61    +0.17± 0.97       -0.26± 0.47      15/15/15/15/15
    all  108    105.34    104.69    105.61    105.37    105.00    +0.65± 0.28       -0.27± 0.25      105/105/105/105/105
-- T2, setting a
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    107.17    104.63    110.24    106.84    106.33    +2.53± 1.64       -3.08± 2.25      3/3/3/3/3
    0.5    3    107.17    105.07    111.23    107.15    110.08    +2.10± 1.91       -4.07± 2.76      3/3/3/3/3
      1    3    107.27    104.43    111.42    107.38    109.07    +2.84± 1.59       -4.15± 2.59      3/3/3/3/3
    1.5    3    105.44    105.76    108.84    105.70    108.44    -0.32± 1.40       -3.40± 3.11      3/3/3/3/3
      2    3    107.31    104.13    110.08    108.48    108.73    +3.18± 3.01       -2.77± 2.61      3/3/3/3/3
    2.5    3    104.32    107.26    106.47    109.33    104.77    -2.94± 1.12       -2.15± 1.65      3/3/3/3/3
    all   18    106.44    105.21    109.71    107.48    107.90    +1.23± 0.83       -3.27± 0.89      18/18/18/18/18
-- T2, setting b
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3     84.75     84.75     84.75     84.75     84.75    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3     78.06     78.06     78.06     78.06     78.06    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3     71.01     71.01     71.01     71.01     71.01    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3     62.39     62.39     62.39     62.39     62.39    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3     53.41     53.41     53.41     53.41     53.41    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3     42.77     42.77     42.77     42.77     42.77    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    all   18     65.40     65.40     65.40     65.40     65.40    +0.00± 0.00       +0.00± 0.00      18/18/18/18/18
-- T2, setting c
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    117.51    117.51    117.51    117.51    117.51    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    111.22    111.22    111.22    111.22    111.22    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3     95.29     95.29     95.29     95.29     95.29    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3     74.50     74.50     74.50     74.50     74.50    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3     49.93     49.93     49.93     49.93     49.93    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    all   18     74.74     74.74     74.74     74.74     74.74    +0.00± 0.00       +0.00± 0.00      15/15/15/15/15
-- T2, setting d
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    129.06    129.06    129.06    129.06    129.06    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    125.63    125.63    125.63    125.63    125.63    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3    117.95    117.95    117.95    117.95    117.95    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3    108.74    108.74    108.74    108.74    108.74    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3     94.69     94.69     94.69     94.69     94.69    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3     81.65     81.65     81.65     81.65     81.65    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    all   18    109.62    109.62    109.62    109.62    109.62    +0.00± 0.00       +0.00± 0.00      18/18/18/18/18
-- T2, setting e
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    158.62    158.62    158.62    158.62    158.62    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    155.20    155.20    155.20    155.20    155.20    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3    143.46    143.46    143.46    143.46    143.46    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3    137.38    137.38    137.38    137.38    137.38    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3    127.31    127.31    127.31    127.31    127.31    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3    110.35    110.35    110.35    110.35    110.35    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    all   18    138.72    138.72    138.72    138.72    138.72    +0.00± 0.00       +0.00± 0.00      18/18/18/18/18
-- T2, setting f
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    148.63    144.33    147.89    146.67    146.28    +4.29± 4.31       +0.74± 0.23      3/3/3/3/3
    0.5    3    147.94    143.40    147.09    145.38    144.94    +4.54± 2.08       +0.85± 3.00      3/3/3/3/3
      1    3    144.80    143.21    142.09    143.02    140.89    +1.59± 3.84       +2.71± 1.13      3/3/3/3/3
    1.5    3    133.96    133.89    129.85    132.11    130.11    +0.07± 4.03       +4.11± 4.04      3/3/3/3/3
      2    3    126.74    125.36    125.83    130.55    125.27    +1.38± 1.10       +0.90± 2.42      3/3/3/3/3
    2.5    3    120.49    116.52    119.92    119.92    114.14    +3.98± 5.88       +0.57± 2.50      3/3/3/3/3
    all   18    137.09    134.45    135.45    136.27    133.60    +2.64± 1.39       +1.65± 0.93      18/18/18/18/18
-- P: mean served streams 8.33, group-common served 1.00, private served 6.33; BS-specific in 0/108 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- J: stream-set test made in 36/108 runs, 1-layer kept in 12
-- CPU per run (s): P 54.8, P1 48.3, J 68.4, JH 74.3, J1 50.6; P discrete design 0.001

===== scenario E3, Q = 50: 108 common runs (P: 108, P1: 108, JH: 108, J1: 108)
-- E3, all settings
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0   18    130.19    129.86    139.50    139.60    139.09    +0.33± 1.34       -9.31± 5.41      18/18/18/18/18
    0.5   18    127.95    130.56    133.02    133.37    132.32    -2.61± 1.26       -5.07± 5.23      18/18/18/18/18
      1   18    124.48    125.05    124.99    126.26    124.38    -0.57± 0.70       -0.51± 5.19      18/18/18/18/18
    1.5   18    118.64    115.75    112.79    112.47    112.00    +2.90± 1.50       +5.86± 4.30      18/18/18/18/18
      2   18    113.67    107.45     95.46     95.46     95.61    +6.21± 2.10      +18.21± 5.57      18/18/16/16/16
    2.5   18     99.24     93.83     78.43     79.12     78.16    +5.41± 4.25      +20.81± 5.99      16/15/12/12/12
    all  108    119.03    117.08    114.03    114.38    113.59    +1.94± 0.92       +5.00± 2.38      106/105/100/100/100
-- E3, setting a
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    137.13    137.13    137.13    137.13    137.13    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    0.5    3    135.09    135.09    135.09    135.09    135.09    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      1    3    143.92    143.92    143.92    143.92    143.92    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    1.5    3    138.21    138.21    138.21    138.21    138.21    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
      2    3    143.74    143.74    143.74    143.74    143.74    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    2.5    3    138.79    138.79    138.79    138.79    138.79    +0.00± 0.00       +0.00± 0.00      3/3/3/3/3
    all   18    139.48    139.48    139.48    139.48    139.48    +0.00± 0.00       +0.00± 0.00      18/18/18/18/18
-- E3, setting b
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3     87.30     86.64     76.67     76.67     76.67    +0.66± 0.31      +10.63± 2.43      3/3/3/3/3
    0.5    3     85.82     84.64     68.77     68.77     68.77    +1.17± 0.40      +17.04± 1.43      3/3/3/3/3
      1    3     76.73     76.93     60.77     60.77     60.77    -0.20± 0.97      +15.96± 0.20      3/3/3/3/3
    1.5    3     67.30     66.19     51.52     51.52     51.52    +1.10± 1.89      +15.78± 3.42      3/3/3/3/3
      2    3     58.79     54.88     42.80     42.80     42.80    +3.91± 4.03      +15.98± 4.58      3/3/3/3/3
    2.5    3     52.63     41.27      0.00      0.00      0.00   +11.36± 5.76      +52.63± 5.01      3/3/0/0/0
    all   18     71.43     68.43     50.09     50.09     50.09    +3.00± 1.41      +21.34± 3.61      18/18/15/15/15
-- E3, setting c
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    106.90    108.50    122.35    122.35    122.35    -1.60± 4.64      -15.45± 2.50      3/3/3/3/3
    0.5    3     98.25    107.21    105.73    105.73    105.73    -8.96± 1.23       -7.48± 4.23      3/3/3/3/3
      1    3     94.06     94.00     85.46     85.46     85.46    +0.06± 2.33       +8.60± 5.83      3/3/3/3/3
    1.5    3     83.79     74.74     64.42     64.42     64.42    +9.05± 1.71      +19.37± 3.22      3/3/3/3/3
      2    3     71.98     50.38     12.73     12.73     12.73   +21.60± 4.09      +59.25± 8.76      3/3/1/1/1
    2.5    3     24.08      0.00      0.00      0.00      0.00   +24.08±24.08      +24.08±24.08      1/0/0/0/0
    all   18     79.84     72.47     65.12     65.12     65.12    +7.37± 4.58      +14.73± 6.99      16/15/13/13/13
-- E3, setting d
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    125.99    127.62    124.94    124.94    124.94    -1.63± 2.94       +1.06± 5.32      3/3/3/3/3
    0.5    3    128.02    129.62    116.46    116.46    116.46    -1.60± 1.54      +11.56± 3.00      3/3/3/3/3
      1    3    125.29    123.00    107.64    107.64    107.64    +2.30± 2.72      +17.65± 4.88      3/3/3/3/3
    1.5    3    118.08    122.04     95.37     95.37     95.37    -3.96± 4.14      +22.71± 4.85      3/3/3/3/3
      2    3    114.72    113.60     81.89     81.89     81.89    +1.13± 1.49      +32.84± 4.04      3/3/3/3/3
    2.5    3    104.27    103.22     67.18     67.18     67.18    +1.05± 3.17      +37.10± 2.86      3/3/3/3/3
    all   18    119.40    119.85     98.91     98.91     98.91    -0.45± 1.10      +20.49± 3.31      18/18/18/18/18
-- E3, setting e
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    140.69    135.57    195.79    195.79    195.79    +5.12± 5.07      -55.10± 1.79      3/3/3/3/3
    0.5    3    143.26    141.71    191.44    191.44    191.44    +1.55± 3.43      -48.18± 2.36      3/3/3/3/3
      1    3    129.20    130.99    173.39    173.39    173.39    -1.79± 0.70      -44.19± 3.45      3/3/3/3/3
    1.5    3    125.74    123.63    150.58    150.58    150.58    +2.10± 1.49      -24.84± 9.18      3/3/3/3/3
      2    3    122.79    119.61    124.31    124.31    124.31    +3.18± 5.41       -1.51± 5.48      3/3/3/3/3
    2.5    3    110.46    113.27     98.11     98.11     98.11    -2.81± 3.62      +12.35± 7.98      3/3/3/3/3
    all   18    128.69    127.46    155.60    155.60    155.60    +1.23± 1.44      -26.91± 6.37      18/18/18/18/18
-- E3, setting f
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    3    183.13    183.69    180.11    180.70    177.67    -0.56± 4.50       +3.02± 3.02      3/3/3/3/3
    0.5    3    177.27    185.09    180.64    182.69    176.42    -7.82± 3.36       -3.37± 5.63      3/3/3/3/3
      1    3    177.69    181.49    178.75    186.37    175.12    -3.79± 0.81       -1.06± 2.85      3/3/3/3/3
    1.5    3    178.75    169.66    176.63    174.75    171.90    +9.09± 4.64       +2.12± 5.79      3/3/3/3/3
      2    3    169.97    162.51    167.27    167.27    168.19    +7.46± 1.50       +2.71± 2.39      3/3/3/3/3
    2.5    3    165.20    166.44    166.51    170.67    164.90    -1.24± 3.73       -1.32± 3.67      3/3/3/3/3
    all   18    175.34    174.81    174.98    177.07    172.37    +0.52± 1.86       +0.35± 1.53      18/18/18/18/18
-- P: mean served streams 10.89, group-common served 1.56, private served 8.33; BS-specific in 72/108 runs; per run, carried by a subset of the BSs: group-common 1.39, private 4.50
-- J: stream-set test made in 18/108 runs, 1-layer kept in 9
-- CPU per run (s): P 287.0, P1 144.8, J 98.0, JH 87.3, J1 99.9; P discrete design 0.002

```

### R1, screen (Q = 5), with --by-setting

`python3 tools/report.py P --budget 5 --by-setting --json`

```

===== scenario T2, Q = 5: 144 common runs (P: 144, P1: 144, JH: 144, J1: 144)
-- T2, all settings
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0   24    104.41    104.65    104.77    104.76    104.41    -0.24± 0.24       -0.36± 0.30      24/24/24/24/24
    0.5   24    102.16    102.40    102.52    102.51    102.16    -0.24± 0.24       -0.36± 0.30      24/24/24/24/24
      1   24     95.34     95.39     95.64     95.63     95.34    -0.05± 0.05       -0.30± 0.29      24/24/24/24/24
    1.5   24     87.03     86.95     87.23     87.23     86.94    +0.08± 0.08       -0.20± 0.21      24/24/24/24/24
      2   24     70.47     69.43     70.13     70.40     69.41    +1.04± 0.61       +0.34± 0.32      20/20/20/20/20
    2.5   24     55.94     54.37     55.88     55.88     54.42    +1.57± 1.02       +0.06± 0.03      16/16/16/16/16
    all  144     85.89     85.53     86.03     86.07     85.45    +0.36± 0.21       -0.14± 0.11      132/132/132/132/132
-- T2, setting a
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    0.5    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
      1    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    1.5    4     93.81     93.80     93.82     93.82     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
      2    4     93.80     93.80     93.81     93.81     93.81    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    2.5    4     93.55     93.55     93.56     93.56     93.56    +0.01± 0.01       -0.01± 0.01      4/4/4/4/4
    all   24     93.76     93.76     93.77     93.77     93.77    +0.01± 0.00       -0.01± 0.00      24/24/24/24/24
-- T2, setting b
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4     73.49     73.49     73.49     73.49     73.49    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4     67.64     67.64     67.64     67.64     67.64    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4     59.67     59.67     59.67     59.67     59.67    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4     50.97     50.97     50.97     50.97     50.97    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4     42.45     42.45     42.45     42.45     42.45    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    all   24     49.04     49.04     49.04     49.04     49.04    +0.00± 0.00       +0.00± 0.00      20/20/20/20/20
-- T2, setting c
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4     96.70     96.70     96.70     96.70     96.70    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4     93.91     93.91     93.91     93.91     93.91    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4     79.27     79.27     79.27     79.27     79.27    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4     58.57     58.57     58.57     58.57     58.57    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    2.5    4      0.00      0.00      0.00      0.00      0.00    +0.00± 0.00       +0.00± 0.00      0/0/0/0/0
    all   24     54.74     54.74     54.74     54.74     54.74    +0.00± 0.00       +0.00± 0.00      16/16/16/16/16
-- T2, setting d
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    113.38    113.38    113.38    113.38    113.38    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    109.01    109.01    109.01    109.01    109.01    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4     99.85     99.85     99.85     99.85     99.85    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4     88.14     88.14     88.14     88.14     88.14    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4     74.76     74.76     74.76     74.76     74.76    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4     60.94     60.94     60.94     60.94     60.94    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24     91.01     91.01     91.01     91.01     91.01    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- T2, setting e
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    133.91    133.91    133.91    133.91    133.91    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    133.43    133.43    133.43    133.43    133.43    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4    124.60    124.60    124.60    124.60    124.60    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4    116.82    116.82    116.82    116.82    116.82    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4    101.53    101.53    101.53    101.53    101.53    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4     80.51     80.51     80.51     80.51     80.51    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24    115.13    115.13    115.13    115.13    115.13    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- T2, setting f
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    115.15    116.60    117.30    117.27    115.16    -1.44± 1.44       -2.14± 1.66      4/4/4/4/4
    0.5    4    115.15    116.60    117.30    117.27    115.16    -1.44± 1.44       -2.14± 1.66      4/4/4/4/4
      1    4    114.84    115.14    116.61    116.58    114.85    -0.30± 0.29       -1.77± 1.74      4/4/4/4/4
    1.5    4    113.89    113.40    115.09    115.08    113.33    +0.49± 0.44       -1.20± 1.25      4/4/4/4/4
      2    4    110.26    104.01    108.19    109.85    103.91    +6.25± 2.51       +2.07± 1.91      4/4/4/4/4
    2.5    4    100.61     91.23    100.26    100.26     91.50    +9.39± 4.87       +0.35± 0.12      4/4/4/4/4
    all   24    111.65    109.50    112.46    112.72    108.99    +2.16± 1.23       -0.80± 0.63      24/24/24/24/24
-- P: mean served streams 8.33, group-common served 1.00, private served 6.33; BS-specific in 0/144 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- J: stream-set test made in 48/144 runs, 1-layer kept in 16
-- CPU per run (s): P 5.0, P1 4.6, J 8.1, JH 6.5, J1 4.4; P discrete design 0.001

===== scenario E3, Q = 5: 144 common runs (P: 144, P1: 144, JH: 144, J1: 144)
-- E3, all settings
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0   24    121.87    121.96    128.91    128.82    128.91    -0.09± 0.08       -7.03± 3.83      24/24/24/24/24
    0.5   24    120.05    120.66    122.90    122.83    122.90    -0.62± 0.77       -2.85± 3.75      24/24/24/24/24
      1   24    115.50    114.71    113.25    113.24    113.25    +0.79± 0.60       +2.26± 3.67      24/24/24/24/24
    1.5   24    108.78    106.03    101.61    101.60    101.71    +2.75± 1.22       +7.17± 3.20      24/24/24/24/24
      2   24    101.67     94.77     83.68     83.68     83.66    +6.91± 2.78      +18.00± 4.91      24/23/20/20/20
    2.5   24     91.12     83.67     70.64     70.64     70.63    +7.45± 3.13      +20.48± 4.39      22/20/16/16/16
    all  144    109.83    106.97    103.50    103.47    103.51    +2.86± 0.78       +6.34± 1.81      142/139/132/132/132
-- E3, setting a
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    125.65    125.65    125.65    125.65    125.65    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    0.5    4    125.56    125.56    125.56    125.56    125.56    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      1    4    125.42    125.42    125.42    125.42    125.42    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    1.5    4    125.25    125.25    125.25    125.25    125.25    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
      2    4    125.04    125.04    125.04    125.04    125.04    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    2.5    4    125.14    125.14    125.14    125.14    125.14    +0.00± 0.00       +0.00± 0.00      4/4/4/4/4
    all   24    125.34    125.34    125.34    125.34    125.34    +0.00± 0.00       +0.00± 0.00      24/24/24/24/24
-- E3, setting b
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4     84.98     84.98     72.05     72.05     72.05    +0.00± 0.00      +12.94± 1.65      4/4/4/4/4
    0.5    4     80.17     80.18     64.42     64.42     64.42    -0.01± 0.01      +15.75± 1.18      4/4/4/4/4
      1    4     71.33     71.07     55.60     55.60     55.60    +0.26± 0.28      +15.73± 0.33      4/4/4/4/4
    1.5    4     62.44     61.28     46.85     46.85     46.85    +1.16± 1.15      +15.60± 0.94      4/4/4/4/4
      2    4     54.10     49.38     38.17     38.17     38.17    +4.73± 2.88      +15.93± 2.14      4/4/4/4/4
    2.5    4     43.13     37.79      0.00      0.00      0.00    +5.34± 3.22      +43.13± 4.13      4/4/0/0/0
    all   24     66.03     64.11     46.18     46.18     46.18    +1.91± 0.81      +19.84± 2.31      24/24/20/20/20
-- E3, setting c
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    100.83    100.83    113.87    113.87    113.87    +0.00± 0.00      -13.04± 3.10      4/4/4/4/4
    0.5    4     94.10     98.46     97.36     97.36     97.36    -4.36± 4.52       -3.26± 4.36      4/4/4/4/4
      1    4     86.23     81.19     76.29     76.29     76.29    +5.04± 2.40       +9.94± 2.34      4/4/4/4/4
    1.5    4     74.21     59.21     54.52     54.52     54.52   +15.00± 1.82      +19.68± 0.29      4/4/4/4/4
      2    4     61.10     28.47      0.00      0.00      0.00   +32.63± 7.42      +61.10± 2.50      4/3/0/0/0
    2.5    4     26.01      0.00      0.00      0.00      0.00   +26.01±15.09      +26.01±15.09      2/0/0/0/0
    all   24     73.74     61.36     57.01     57.01     57.01   +12.39± 3.83      +16.74± 5.52      22/19/16/16/16
-- E3, setting d
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    116.89    116.89    115.43    115.43    115.43    -0.00± 0.00       +1.47± 4.49      4/4/4/4/4
    0.5    4    118.42    118.42    109.73    109.73    109.73    -0.00± 0.00       +8.70± 4.84      4/4/4/4/4
      1    4    118.80    118.81     99.18     99.18     99.18    -0.01± 0.01      +19.63± 4.30      4/4/4/4/4
    1.5    4    110.68    111.42     86.12     86.12     86.12    -0.74± 0.58      +24.57± 4.11      4/4/4/4/4
      2    4    103.48    103.11     72.35     72.35     72.35    +0.37± 0.33      +31.13± 3.27      4/4/4/4/4
    2.5    4     94.01     91.26     58.64     58.64     58.64    +2.75± 1.67      +35.37± 4.35      4/4/4/4/4
    all   24    110.38    109.99     90.24     90.24     90.24    +0.40± 0.35      +20.14± 2.92      24/24/24/24/24
-- E3, setting e
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    136.21    136.21    179.41    179.41    179.41    +0.00± 0.00      -43.20± 2.63      4/4/4/4/4
    0.5    4    135.42    134.36    173.48    173.48    173.48    +1.06± 0.87      -38.06± 5.01      4/4/4/4/4
      1    4    125.57    126.06    157.45    157.45    157.45    -0.49± 1.82      -31.88± 4.77      4/4/4/4/4
    1.5    4    117.63    116.55    135.06    135.06    135.06    +1.08± 1.59      -17.43± 6.65      4/4/4/4/4
      2    4    110.06    106.34    110.21    110.21    110.21    +3.72± 4.44       -0.15± 9.16      4/4/4/4/4
    2.5    4    103.12     92.63     84.76     84.76     84.76   +10.48± 6.72      +18.36± 9.33      4/4/4/4/4
    all   24    121.33    118.69    140.06    140.06    140.06    +2.64± 1.47      -18.73± 5.16      24/24/24/24/24
-- E3, setting f
   R_th    n         P        P1         J        JH        J1              P-P1               P-J   QoS met P/P1/J/JH/J1
      0    4    166.66    167.20    167.03    166.53    167.04    -0.54± 0.50       -0.37± 0.58      4/4/4/4/4
    0.5    4    166.61    166.99    166.86    166.43    166.87    -0.39± 0.34       -0.25± 0.47      4/4/4/4/4
      1    4    165.67    165.70    165.53    165.51    165.54    -0.03± 0.02       +0.14± 0.15      4/4/4/4/4
    1.5    4    162.45    162.47    161.85    161.82    162.46    -0.02± 0.03       +0.61± 0.58      4/4/4/4/4
      2    4    156.27    156.27    156.28    156.28    156.22    -0.00± 0.02       -0.01± 0.01      4/4/4/4/4
    2.5    4    155.31    155.22    155.32    155.32    155.25    +0.09± 0.07       -0.01± 0.00      4/4/4/4/4
    all   24    162.16    162.31    162.15    161.98    162.23    -0.15± 0.10       +0.02± 0.16      24/24/24/24/24
-- P: mean served streams 10.96, group-common served 1.62, private served 8.33; BS-specific in 96/144 runs; per run, carried by a subset of the BSs: group-common 1.46, private 4.50
-- J: stream-set test made in 24/144 runs, 1-layer kept in 13
-- CPU per run (s): P 35.0, P1 16.7, J 14.2, JH 9.2, J1 10.4; P discrete design 0.002

```

### R2, R3: benchmark setting

`python3 tools/report.py P --bench --json`

```

===== benchmark setting (T2, K = 16, N = 3, S_max = 12, R_th = 2, Q = 50): 15 common runs (P: 15, sparse_bf: 15, P@SC: 15, J: 15, J@SC: 15)
sparse_bf, J (the current framework) and J@SC (stream clustering on its streams): stored runs of Fig. 5.10
-- by M
      M    n         P sparse_bf      P@SC         J      J@SC       P-sparse_bf            P-P@SC   QoS met P/sparse_bf/P@SC/J/J@SC
    100    5     74.68     63.08     44.18     73.51     44.18   +11.61± 3.43      +30.51± 2.05      5/5/5/5/5
    300    5     81.44     75.09     54.38     81.95     54.38    +6.35± 6.04      +27.06± 1.62      5/5/5/5/5
    500    5     85.68     81.03     64.51     88.72     64.51    +4.65± 4.33      +21.17± 3.84      5/5/5/5/5
    all   15     80.60     73.07     54.36     81.39     54.36    +7.54± 2.65      +26.25± 1.76      15/15/15/15/15
-- P: mean served streams 11.40, group-common served 4.40, private served 6.00; BS-specific in 0/15 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- P: shares of the sum rate over its 15 runs meeting the QoS: super-common 15.6%, group-common 9.2%, private 75.1%

```

### R4: sum rate versus S_max

`python3 tools/report.py P --smax --json`

```

===== sum rate versus S_max (T2, N = 3, M = 300, Q = 50): 126 runs of P
-- K = 8: mean sum rate over the drops (QoS met / runs)
  R_th          S=4          S=6          S=8         S=10         S=12         S=14         S=16
     0   83.82  2/2   101.91  2/2   105.07  2/2   102.22  2/2   102.22  2/2   102.22  2/2   102.22  2/2 
     1   79.28  2/2    98.44  2/2   105.84  2/2   102.34  2/2   102.34  2/2   102.34  2/2   102.34  2/2 
     2   69.89  2/2    99.06  2/2   104.56  2/2   102.07  2/2   102.07  2/2   102.07  2/2   102.07  2/2 
   all   77.66  6/6    99.80  6/6   105.16  6/6   102.21  6/6   102.21  6/6   102.21  6/6   102.21  6/6 
   change to the next S_max (paired over R_th and drops): 4->6 +22.14±3.48, 6->8 +5.35±2.40, 8->10 -2.95±1.58, 10->12 +0.00±0.00, 12->14 +0.00±0.00, 14->16 +0.00±0.00
-- K = 12: mean sum rate over the drops (QoS met / runs)
  R_th          S=4          S=6          S=8         S=10         S=12         S=14         S=16
     0   84.37  2/2   113.44  2/2   120.67  2/2   125.13  2/2   123.78  2/2   123.78  2/2   123.78  2/2 
     1   68.85  2/2   106.67  2/2   116.12  2/2   119.18  2/2   119.36  2/2   119.36  2/2   119.36  2/2 
     2   51.03  2/2    82.86  2/2    98.83  2/2   105.54  2/2   105.79  2/2   105.79  2/2   105.79  2/2 
   all   68.08  6/6   100.99  6/6   111.87  6/6   116.62  6/6   116.31  6/6   116.31  6/6   116.31  6/6 
   change to the next S_max (paired over R_th and drops): 4->6 +32.91±1.76, 6->8 +10.88±2.54, 8->10 +4.74±2.77, 10->12 -0.30±0.49, 12->14 +0.00±0.00, 14->16 +0.00±0.00
-- K = 16: mean sum rate over the drops (QoS met / runs)
  R_th          S=4          S=6          S=8         S=10         S=12         S=14         S=16
     0   84.23  2/2   116.64  2/2   122.32  2/2   123.59  2/2   122.63  2/2   122.63  2/2   122.63  2/2 
     1   63.37  2/2    94.37  2/2   112.06  2/2   110.32  2/2   115.43  2/2   115.43  2/2   115.43  2/2 
     2    0.00  0/2    50.59  2/2    66.83  2/2    76.74  2/2    91.99  2/2    91.99  2/2    91.99  2/2 
   all   49.20  4/6    87.20  6/6   100.40  6/6   103.55  6/6   110.02  6/6   110.02  6/6   110.02  6/6 
   change to the next S_max (paired over R_th and drops): 4->6 +38.00±3.99, 6->8 +13.20±3.19, 8->10 +3.15±2.54, 10->12 +6.47±4.07, 12->14 +0.00±0.00, 14->16 +0.00±0.00

```

### Reference: the joint design JH against J1 (confirm tier, Q = 50)

`python3 tools/report.py JH --json`

```

===== scenario T2, Q = 50: 108 common runs (JH: 108, J1: 108)
-- T2, all settings
   R_th    n        JH         J        J1             JH-J1              J-J1   QoS met JH/J/J1
      0   18    123.91    124.68    123.76    +0.15± 0.81       +0.92± 0.47      18/18/18
    0.5   18    120.44    121.41    120.85    -0.41± 1.00       +0.55± 0.64      18/18/18
      1   18    113.02    113.54    112.95    +0.07± 0.86       +0.59± 0.46      18/18/18
    1.5   18    103.47    103.61    103.59    -0.12± 0.71       +0.02± 0.27      18/18/18
      2   18     94.06     93.54     93.22    +0.84± 0.63       +0.32± 0.18      18/18/18
    2.5   18     77.34     76.86     75.61    +1.72± 0.97       +1.25± 0.89      15/15/15
    all  108    105.37    105.61    105.00    +0.37± 0.34       +0.61± 0.22      105/105/105
-- JH: mean served streams 9.50, group-common served 1.00, private served 7.50; BS-specific in 0/108 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- J: stream-set test made in 36/108 runs, 1-layer kept in 12
-- CPU per run (s): JH 74.3, J 68.4, J1 50.6; JH discrete design 0.001

===== scenario E3, Q = 50: 108 common runs (JH: 108, J1: 108)
-- E3, all settings
   R_th    n        JH         J        J1             JH-J1              J-J1   QoS met JH/J/J1
      0   18    139.60    139.50    139.09    +0.50± 0.43       +0.41± 0.41      18/18/18
    0.5   18    133.37    133.02    132.32    +1.05± 0.76       +0.70± 0.70      18/18/18
      1   18    126.26    124.99    124.38    +1.87± 1.19       +0.60± 0.60      18/18/18
    1.5   18    112.47    112.79    112.00    +0.47± 0.65       +0.79± 0.55      18/18/18
      2   18     95.46     95.46     95.61    -0.15± 0.73       -0.15± 0.73      16/16/16
    2.5   18     79.12     78.43     78.16    +0.96± 0.73       +0.27± 0.27      12/12/12
    all  108    114.38    114.03    113.59    +0.78± 0.32       +0.44± 0.23      100/100/100
-- JH: mean served streams 8.83, group-common served 0.17, private served 7.67; BS-specific in 0/108 runs; per run, carried by a subset of the BSs: group-common 0.00, private 0.00
-- J: stream-set test made in 18/108 runs, 1-layer kept in 9
-- CPU per run (s): JH 87.3, J 98.0, J1 99.9; JH discrete design 0.001

```

### Self-test after the changes of this round

`python3 harness/selftest.py`

```
JH_T2               : identical trace: True  rate 126.9181  [27 CPU-s]
J1_T2               : identical trace: True  rate 128.7293  [18 CPU-s]
JH_E3               : identical trace: True  rate 166.6123  [23 CPU-s]
rand_fronthaul_E3   : identical trace: True  rate 45.1868  [33 CPU-s]
sparse_bf_T2        : identical trace: True  rate 73.6943  [81 CPU-s]
J_T2 (derived)      : equals the thesis framework run with the stream-set test: True
SELFTEST PASSED
```

### Large networks (discrete design only)

`python3 tools/large_network.py`

```

== B = 2 BSs over 100 m x 100 m, K = 16 UEs, N = 3
  S_max= 4: c=  3 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 3 by several); group-common 0 of 3 (BSs each 0.0), non-core private 0; distinct streams 4, per BS 4..4
  S_max= 6: c=  5 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 5 by several); group-common 0 of 3 (BSs each 0.0), non-core private 0; distinct streams 6, per BS 6..6
  S_max= 9: c=  6 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 6 by several); group-common 3 of 4 (BSs each 1.3), non-core private 0; distinct streams 10, per BS 9..9
  S_max=12: c=  6 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 6 by several); group-common 4 of 4 (BSs each 2.0), non-core private 0; distinct streams 11, per BS 11..11
  S_max=24: c=  6 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 6 by several); group-common 4 of 4 (BSs each 2.0), non-core private 0; distinct streams 11, per BS 11..11

== B = 3 BSs over 100 m x 100 m, K = 16 UEs, N = 3
  S_max= 4: c=  4 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 4 by several); group-common 1 of 3 (BSs each 1.0), non-core private 0; distinct streams 6, per BS 4..4
  S_max= 6: c=  6 core UEs, m= 2; BSs per core min/mean/max 2/2.0/2 (0 cores carried by one BS only, 6 by several); group-common 3 of 5 (BSs each 1.0), non-core private 0; distinct streams 10, per BS 6..6
  S_max= 9: c=  8 core UEs, m= 3; BSs per core min/mean/max 3/3.0/3 (0 cores carried by one BS only, 8 by several); group-common 0 of 5 (BSs each 0.0), non-core private 0; distinct streams 9, per BS 9..9
  S_max=12: c=  9 core UEs, m= 3; BSs per core min/mean/max 3/3.0/3 (0 cores carried by one BS only, 9 by several); group-common 5 of 5 (BSs each 1.2), non-core private 0; distinct streams 15, per BS 12..12
  S_max=24: c=  9 core UEs, m= 3; BSs per core min/mean/max 3/3.0/3 (0 cores carried by one BS only, 9 by several); group-common 5 of 5 (BSs each 3.0), non-core private 0; distinct streams 15, per BS 15..15

== B = 9 BSs over 300 m x 300 m, K = 36 UEs, N = 4
  S_max= 5: c= 12 core UEs, m= 3; BSs per core min/mean/max 3/3.0/3 (0 cores carried by one BS only, 12 by several); group-common 0 of 7 (BSs each 0.0), non-core private 0; distinct streams 13, per BS 5..5
  S_max= 8: c= 15 core UEs, m= 4; BSs per core min/mean/max 4/4.0/4 (0 cores carried by one BS only, 15 by several); group-common 3 of 9 (BSs each 1.0), non-core private 0; distinct streams 19, per BS 8..8
  S_max=12: c= 19 core UEs, m= 5; BSs per core min/mean/max 5/5.0/5 (0 cores carried by one BS only, 19 by several); group-common 4 of 9 (BSs each 1.0), non-core private 0; distinct streams 24, per BS 12..12
  S_max=16: c= 22 core UEs, m= 6; BSs per core min/mean/max 3/5.9/6 (0 cores carried by one BS only, 22 by several); group-common 5 of 8 (BSs each 1.2), non-core private 0; distinct streams 28, per BS 16..16
  S_max=32: c= 32 core UEs, m= 8; BSs per core min/mean/max 4/7.9/8 (0 cores carried by one BS only, 32 by several); group-common 3 of 3 (BSs each 4.0), non-core private 0; distinct streams 36, per BS 26..32

== B = 16 BSs over 600 m x 600 m, K = 64 UEs, N = 4
  S_max= 5: c= 16 core UEs, m= 4; BSs per core min/mean/max 4/4.0/4 (0 cores carried by one BS only, 16 by several); group-common 0 of 12 (BSs each 0.0), non-core private 0; distinct streams 17, per BS 5..5
  S_max= 8: c= 20 core UEs, m= 5; BSs per core min/mean/max 5/5.0/5 (0 cores carried by one BS only, 20 by several); group-common 10 of 15 (BSs each 1.2), non-core private 0; distinct streams 31, per BS 8..8
  S_max=12: c= 25 core UEs, m= 7; BSs per core min/mean/max 4/6.9/7 (0 cores carried by one BS only, 25 by several); group-common 4 of 18 (BSs each 1.0), non-core private 0; distinct streams 30, per BS 12..12
  S_max=16: c= 30 core UEs, m= 8; BSs per core min/mean/max 4/7.7/8 (0 cores carried by one BS only, 30 by several); group-common 8 of 19 (BSs each 1.0), non-core private 0; distinct streams 39, per BS 16..16
  S_max=32: c= 44 core UEs, m=11; BSs per core min/mean/max 8/10.8/11 (0 cores carried by one BS only, 44 by several); group-common 15 of 17 (BSs each 1.3), non-core private 0; distinct streams 60, per BS 32..32

== B = 25 BSs over 1000 m x 1000 m, K = 100 UEs, N = 4
  S_max= 5: c= 20 core UEs, m= 5; BSs per core min/mean/max 2/4.8/5 (0 cores carried by one BS only, 20 by several); group-common 3 of 18 (BSs each 1.0), non-core private 0; distinct streams 24, per BS 5..5
  S_max= 8: c= 25 core UEs, m= 7; BSs per core min/mean/max 6/7.0/7 (0 cores carried by one BS only, 25 by several); group-common 1 of 20 (BSs each 1.0), non-core private 0; distinct streams 27, per BS 8..8
  S_max=12: c= 32 core UEs, m= 8; BSs per core min/mean/max 8/8.0/8 (0 cores carried by one BS only, 32 by several); group-common 10 of 21 (BSs each 1.9), non-core private 0; distinct streams 43, per BS 12..12
  S_max=16: c= 37 core UEs, m=10; BSs per core min/mean/max 7/9.8/10 (0 cores carried by one BS only, 37 by several); group-common 10 of 23 (BSs each 1.1), non-core private 0; distinct streams 48, per BS 16..16
  S_max=32: c= 55 core UEs, m=14; BSs per core min/mean/max 7/13.5/14 (0 cores carried by one BS only, 55 by several); group-common 20 of 23 (BSs each 1.6), non-core private 0; distinct streams 76, per BS 32..32
```



