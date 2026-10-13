"""Discrete-design variants for the harness: name -> entry.

Entry keys:
  design      'module:function' in code/ ; the function design(net, p) -> hrs.Design (see code/discrete.py)
  one_layer   the name of the variant that is this design's 1-layer RSMA counterpart (requirement R1 of BRIEF.md)
  opt         (optional) dict of options passed to discrete.run (keys of newprop.NEW); leave it out unless your design
              needs a changed evaluation, and then explain why in DESIGN.md
Use a new name instead of editing a variant that already has results.
KEY@SC (no entry needed; harness/grid.py and tools/report.py) is the stream-clustering benchmark (code/benchmarks.py)
on the streams of KEY's design.

The baselines (the current thesis):
  JH   current design (joint transmission, HRS order)           discrete.current
  J1   its 1-layer RSMA (group-common streams removed)          discrete.current_1layer
  J    the thesis framework = JH or J1 by the stream-set test at the initial design of the search; derived by
       tools/report.py from the runs of JH and J1 (not run separately)
"""

VARIANTS = {
    'JH': dict(design='discrete:current', one_layer='J1'),
    'J1': dict(design='discrete:current_1layer'),
    'V1': dict(design='perbs:perbs_rank', one_layer='V11'),
    'V11': dict(design='perbs:perbs_rank_1l'),
    'C1': dict(design='perbs_v:c1', one_layer='C11'),
    'C11': dict(design='perbs_v:c1_1l'),
    'C2': dict(design='perbs_v:c2', one_layer='C21'),
    'C21': dict(design='perbs_v:c2_1l'),
    'C3': dict(design='perbs_v:c3', one_layer='C31'),
    'C31': dict(design='perbs_v:c3_1l'),
    'V1b': dict(design='perbs:perbs_rank', one_layer='V1b1', opt=dict(init='core_bs', seed='rule_bs')),
    'V1b1': dict(design='perbs:perbs_rank_1l', opt=dict(init='core_bs', seed='rule_bs')),
    'F1': dict(design='perbs:perbs_feas', one_layer='V11'),
}
