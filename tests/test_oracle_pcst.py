"""Unabhängiges Orakel: das exakte Optimum (Dreyfus-Wagner) gegen (1) Aufzählung ALLER Kantenmengen auf winzigen Graphen mit Gleichständen und verschiedenen Erlösen und (2) ein ganzzahliges
Programm (Mehr-Güter-Fluss, HiGHS über scipy.optimize.milp) auf Stadtplänen; dazu die Goemans-Williamson-Untergrenze (<= Optimum), die Garantie gegen diese Untergrenze und die Optimalität des
starken Beschneidens unter allen Teilbäumen des Rohbaums."""

import math
import random
from dataclasses import dataclass

import numpy as np
import pytest

import pcst_algorithm as A
import pcst_scenario as S


@dataclass
class Mini:
    n: int
    edges: tuple
    depot: int
    customers: tuple
    prizes: tuple


def _tree_nodes(sel, depot):
    """Knotenmenge, wenn `sel` ein Baum durch das Depot ist, sonst None."""
    nodes = {x for e in sel for x in e[:2]}
    if depot not in nodes or len(sel) != len(nodes) - 1:
        return None
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for u, v, _w in sel:
        a, b = find(u), find(v)
        if a == b:
            return None
        parent[a] = b
    return nodes


def _brute(inst):
    best = math.fsum(inst.prizes[c] for c in inst.customers)          # leerer Baum
    edges = list(inst.edges)
    for mask in range(1, 1 << len(edges)):
        sel = [edges[i] for i in range(len(edges)) if mask >> i & 1]
        nodes = _tree_nodes(sel, inst.depot)
        if nodes is not None:
            best = min(best, sum(e[2] for e in sel) + math.fsum(inst.prizes[c] for c in inst.customers if c not in nodes))
    return best


def _milp(inst):
    opt = pytest.importorskip("scipy.optimize")
    from scipy.sparse import lil_matrix

    n, depot, customers = inst.n, inst.depot, list(inst.customers)
    arcs = [(u, v, w) for u, v, w in inst.edges] + [(v, u, w) for u, v, w in inst.edges]
    na, K = len(arcs), len(customers)
    nv = na + n + K * na                                           # Bögen, Knotenwahl z, Fluss je Kunde
    c = np.zeros(nv)
    for a, (_u, _v, w) in enumerate(arcs):
        c[a] = w
    const = 0.0
    for v in customers:
        c[na + v] = -inst.prizes[v]
        const += inst.prizes[v]
    rows = []

    def add(coefs, lo, hi):
        rows.append((coefs, lo, hi))

    for v in range(n):
        co = {a: 1 for a, (_x, y, _w) in enumerate(arcs) if y == v}
        if v != depot:
            co[na + v] = -1
        add(co, 0, 0)
    for a, (x, _y, _w) in enumerate(arcs):
        add({a: 1, na + x: -1}, -np.inf, 0)
    for k, cu in enumerate(customers):
        base = na + n + k * na
        for v in range(n):
            co = {}
            for a, (x, y, _w) in enumerate(arcs):
                if x == v:
                    co[base + a] = co.get(base + a, 0) + 1
                if y == v:
                    co[base + a] = co.get(base + a, 0) - 1
            co[na + cu] = co.get(na + cu, 0) + (-1 if v == depot else (1 if v == cu else 0))
            add(co, 0, 0)
        for a in range(na):
            add({base + a: 1, a: -1}, -np.inf, 0)
    mat = lil_matrix((len(rows), nv))
    for r, (co, _lo, _hi) in enumerate(rows):
        for j, val in co.items():
            mat[r, j] = val
    integrality = np.zeros(nv)
    integrality[:na + n] = 1
    lb, ub = np.zeros(nv), np.ones(nv)
    lb[na + depot] = 1
    res = opt.milp(c, constraints=opt.LinearConstraint(mat.tocsr(), [r[1] for r in rows], [r[2] for r in rows]), integrality=integrality, bounds=opt.Bounds(lb, ub))
    assert res.status == 0
    return float(res.fun) + const


def _tiny(rng):
    while True:
        n = rng.randint(3, 6)
        perm = list(range(n))
        rng.shuffle(perm)
        es = {(min(a, b), max(a, b)) for a, b in zip(perm, perm[1:])}
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        es |= set(rng.sample(pairs, rng.randint(0, 3)))
        if len(es) > 9:
            continue
        wmax = rng.choice([1, 2, 5])
        edges = tuple((u, v, float(rng.randint(1, wmax))) for u, v in sorted(es))
        depot = rng.randrange(n)
        others = [x for x in range(n) if x != depot]
        customers = tuple(sorted(rng.sample(others, rng.randint(1, min(4, len(others))))))
        prizes = [0.0] * n
        mode = rng.choice(["int", "float", "equal", "zero"])
        for c in customers:
            prizes[c] = float(rng.randint(0, 10)) if mode == "int" else round(rng.uniform(0, 10), 2) if mode == "float" else (3.0 if mode == "equal" else 0.0)
        return Mini(n, edges, depot, customers, tuple(prizes))


def test_exact_gw_bound_guarantee_and_strong_prune_against_edge_enumeration():
    rng = random.Random(13)
    for _ in range(60):
        inst = _tiny(rng)
        g = A.Graph(inst)
        opt = _brute(inst)
        ex = A.pcst_exact(g)
        assert ex.solution.objective == pytest.approx(opt) and ex.value == pytest.approx(opt)
        res, sols = A.gw_solutions(g)
        assert res.dual <= opt + 1e-6
        bound = 2.0 - 1.0 / (inst.n - 1)
        for name in ("gw", "gws"):
            assert opt - 1e-6 <= sols[name].objective <= bound * res.dual + 1e-6
        assert sols["gws"].objective <= sols["gw"].objective + 1e-9
        tree = res.root_edges                                       # starkes Beschneiden: bester Teilbaum durch das Depot
        w = {(u, v): c for u, v, c in inst.edges}
        best = math.inf
        for mask in range(1 << len(tree)):
            sel = [(u, v, w[(u, v)]) for i, (u, v) in enumerate(tree) if mask >> i & 1]
            nodes = _tree_nodes(sel, inst.depot) if sel else {inst.depot}
            if nodes is not None:
                best = min(best, sum(e[2] for e in sel) + math.fsum(inst.prizes[c] for c in inst.customers if c not in nodes))
        assert sols["gws"].objective == pytest.approx(best)


@pytest.mark.parametrize("seed", range(8))
def test_city_instances_against_the_integer_program(seed):
    rng = random.Random(seed)
    inst = S.generate(rng.choice([4, 5]), rng.randint(2, 6), rng.choice([0.0, 0.1, 0.3]), rng.choice(["uniform", "clusters"]), rng.choice(["equal", "mixed"]),
                      rng.choice([0.5, 1.5, 2.5, 4.0, 8.0]), seed)
    opt = _milp(inst)
    g = A.Graph(inst)
    assert A.pcst_exact(g).solution.objective == pytest.approx(opt)
    res, sols = A.gw_solutions(g)
    assert res.dual <= opt + 1e-6
    for s in (*sols.values(), A.steiner_all(g), A.connect_none(g)):
        assert s.objective >= opt - 1e-6
