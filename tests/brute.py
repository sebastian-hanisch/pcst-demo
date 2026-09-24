"""Unabhängige Referenzen für die Tests: Zielwert-Brute-Force über alle Baumkantenmengen und über alle Knotenmengen, eigene Union-Find-Funktion, Zufallsinstanzen mit Gleichständen."""

import itertools
import math

import numpy as np

import pcst_scenario as S


def find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def random_instance(rng, n=7, extra=3, t=4, wmax=4, pmax=6, integer=True):
    """Zusammenhängender Zufallsgraph (Zufallsbaum + `extra` weitere Kanten), Kantenlängen 1..wmax (ganzzahlig: viele Gleichstände), Depot 0, t Kunden mit Erlös 0..pmax."""
    pairs = {}
    for v in range(1, n):
        u = int(rng.integers(0, v))
        pairs[(u, v)] = 1
    for _ in range(extra):
        u, v = sorted(int(x) for x in rng.choice(n, size=2, replace=False))
        pairs[(u, v)] = 1
    edges = tuple((u, v, float(rng.integers(1, wmax + 1)) if integer else float(rng.uniform(0.5, wmax))) for u, v in sorted(pairs))
    customers = tuple(sorted(int(x) for x in rng.choice(np.arange(1, n), size=min(t, n - 1), replace=False)))
    prizes = [0.0] * n
    for c in customers:
        prizes[c] = float(rng.integers(0, pmax + 1)) if integer else float(rng.uniform(0, pmax))
    return S.Instance(np.zeros((n, 2)), edges, 0, customers, tuple(prizes), 0, "random")


def objective_of(inst, edges):
    nodes = {inst.depot} | {x for e in edges for x in e}
    w = {(u, v): x for u, v, x in inst.edges}
    return math.fsum(w[e] for e in edges) + math.fsum(inst.prizes[c] for c in inst.customers if c not in nodes)


def brute_trees(inst):
    """Kleinster Zielwert über ALLE Kantenmengen, die einen Baum durch das Depot bilden (leere Menge = leerer Baum). Nur für kleine m (2^m)."""
    edges = [(u, v) for u, v, _w in inst.edges]
    best, best_set = math.inf, None
    for r in range(0, len(edges) + 1):
        for sub in itertools.combinations(edges, r):
            p = list(range(inst.n))
            ok = True
            for u, v in sub:
                a, b = find(p, u), find(p, v)
                if a == b:
                    ok = False
                    break
                p[a] = b
            if not ok:
                continue
            nodes = {x for e in sub for x in e}
            if sub and inst.depot not in nodes:
                continue
            if sub and len({find(p, x) for x in nodes}) != 1:
                continue
            val = objective_of(inst, sub)
            if val < best - 1e-9:
                best, best_set = val, sub
    return best, best_set


def brute_nodesets(inst):
    """Kernsatz-Brute-Force: min über alle Knotenmengen X mit Depot von MST(G[X]) + Erlöse der Kunden außerhalb von X (nur zusammenhängende G[X])."""
    n = inst.n
    others = [v for v in range(n) if v != inst.depot]
    edges = sorted(inst.edges, key=lambda e: (e[2], e[0], e[1]))
    best = math.inf
    for r in range(len(others) + 1):
        for X in itertools.combinations(others, r):
            xs = set(X) | {inst.depot}
            p = list(range(n))
            cost, cnt = 0.0, 0
            for u, v, w in edges:
                if u in xs and v in xs:
                    a, b = find(p, u), find(p, v)
                    if a != b:
                        p[a] = b
                        cost += w
                        cnt += 1
            if cnt != len(xs) - 1:
                continue
            best = min(best, cost + math.fsum(inst.prizes[c] for c in inst.customers if c not in xs))
    return best
