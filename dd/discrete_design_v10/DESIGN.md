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

RESULTS_PLACEHOLDER
