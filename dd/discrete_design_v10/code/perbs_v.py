"""Exploration variants (thin wrappers)."""
import perbs as P
c1 = lambda net, p: P.cell_first(net, p, True, 'local', 'O1')
c1_1l = lambda net, p: P.cell_first(net, p, False, 'local', 'O1')
c2 = lambda net, p: P.cell_first(net, p, True, 'local', 'O2')
c2_1l = lambda net, p: P.cell_first(net, p, False, 'local', 'O2')
c3 = lambda net, p: P.cell_first(net, p, True, 'global', 'O1')
c3_1l = lambda net, p: P.cell_first(net, p, False, 'global', 'O1')
