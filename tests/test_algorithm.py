"""Korrektheitskette der Algorithmen (vor jeder Messung): Kernsatz und Exaktheit gegen Brute-Force, Gültigkeit, starkes Beschneiden, Goemans-Williamson (Dual-Zertifikat, Garantie, Buchführung), Schrankenkette,
Lokalsuche, Sonderfälle."""

import itertools
import math

import numpy as np
import pytest

import pcst_algorithm as A
import pcst_scenario as S
from brute import brute_nodesets, brute_trees, find, objective_of, random_instance

EPS = 1e-7


def _all_solutions(g, res=None):
    res, gw = A.gw_solutions(g) if res is None else res
    sols = dict(gw)
    sols["all"] = A.steiner_all(g)
    sols["stp"] = A.steiner_then_prune(g, sols["all"])
    sols["none"] = A.connect_none(g)
    sols["ls"] = A.local_search(g, min((sols["gws"], sols["stp"]), key=lambda s: s.objective).edges)
    sols["exact"] = A.pcst_exact(g).solution
    return res, sols


# --- 1. Kernsatz und Exaktheit --------------------------------------------------------------------------------------------------------------------


def test_kernel_theorem_and_exactness_on_small_random_graphs():
    rng = np.random.default_rng(7)
    for i in range(160):
        inst = random_instance(rng, n=int(rng.integers(3, 7)), extra=int(rng.integers(0, 4)), t=int(rng.integers(1, 5)), pmax=int(rng.choice([2, 6, 40])))
        g = A.Graph(inst)
        trees, _ = brute_trees(inst)
        nodesets = brute_nodesets(inst)
        ex = A.pcst_exact(g)
        assert trees == pytest.approx(nodesets, abs=EPS), i
        assert ex.value == pytest.approx(trees, abs=EPS), i
        assert ex.solution.objective == pytest.approx(trees, abs=EPS), i


def test_exactness_on_small_city_instances():
    for seed in range(40):
        for blocked, prize_mode, level in ((0.0, "equal", 1.0), (0.3, "mixed", 2.0), (0.2, "equal", 4.0)):
            inst = S.generate(side=3, t=1 + seed % 4, blocked=blocked, prize_mode=prize_mode, level=level, seed=seed)
            g = A.Graph(inst)
            assert A.pcst_exact(g).value == pytest.approx(brute_nodesets(inst), abs=EPS)


def test_zero_prizes_give_the_empty_tree_and_huge_prizes_the_steiner_tree():
    for seed in range(15):
        inst = S.generate(side=3, t=3, blocked=0.1, level=0.0, seed=seed)
        ex = A.pcst_exact(A.Graph(inst))
        assert ex.solution.edges == [] and ex.value == 0.0 and ex.solution.connected == []
        big = S.generate(side=3, t=3, blocked=0.1, level=1e4, seed=seed)
        g = A.Graph(big)
        ex = A.pcst_exact(g)
        assert ex.solution.connected == list(g.customers)
        assert ex.value == pytest.approx(brute_nodesets(big), abs=EPS)
        assert ex.solution.cost == pytest.approx(ex.value, abs=EPS)


def test_single_customer_at_the_threshold_has_equal_objective_either_way():
    inst = S.Instance(np.zeros((3, 2)), ((0, 1, 2.0), (1, 2, 3.0)), 0, (2,), (0.0, 0.0, 5.0), 0)
    g = A.Graph(inst)
    ex = A.pcst_exact(g)
    assert ex.value == pytest.approx(5.0)
    assert A.objective(g, [(0, 1), (1, 2)]) == pytest.approx(5.0) and A.objective(g, []) == pytest.approx(5.0)
    assert ex.solution.edges == []


# --- 2. Gültigkeit --------------------------------------------------------------------------------------------------------------------------------


def _instances(count=30, side=5):
    out = []
    for seed in range(count):
        out.append(S.generate(side=side, t=3 + seed % 6, blocked=(0.0, 0.2, 0.4)[seed % 3], layout=("uniform", "clusters")[seed % 2], prize_mode=("equal", "mixed")[(seed // 2) % 2], level=(1.0, 2.5, 6.0)[seed % 3], seed=seed))
    return out


def test_all_methods_return_valid_trees_with_correct_bookkeeping():
    for inst in _instances(24):
        g = A.Graph(inst)
        _res, sols = _all_solutions(g)
        for name, s in sols.items():
            assert A.is_pcst_solution(g, s.edges), name
            assert s.objective == pytest.approx(objective_of(inst, s.edges), abs=EPS), name
            assert s.cost == pytest.approx(sum(g.w[e] for e in s.edges), abs=EPS)
            assert s.net == pytest.approx(inst.total_prize - s.objective, abs=EPS), name
            assert s.connected == [c for c in g.customers if c in ({g.depot} | {x for e in s.edges for x in e})]
            if name != "gwr":
                deg = {}
                for u, v in s.edges:
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
                assert all(x in g.customers or x == g.depot for x, dg in deg.items() if dg == 1), name


def test_is_pcst_solution_rejects_bad_edge_sets():
    inst = S.textbook_instance()
    g = A.Graph(inst)
    assert A.is_pcst_solution(g, [])
    assert not A.is_pcst_solution(g, [(3, 4)])                      # ohne Depot
    assert not A.is_pcst_solution(g, [(0, 1), (2, 3)])              # nicht zusammenhängend
    assert not A.is_pcst_solution(g, [(0, 4)])                      # keine Kante des Graphen
    assert not A.is_pcst_solution(g, [(3, 4), (3, 5), (4, 5), (2, 3), (1, 2), (0, 1)])   # Kreis


# --- 3. Starkes Beschneiden -----------------------------------------------------------------------------------------------------------------------


def _rooted_subtrees(g, edges):
    """Alle Kantenmengen, die Teilbäume des Baums `edges` durch das Depot sind (Brute-Force)."""
    edges = list(edges)
    out = []
    for r in range(len(edges) + 1):
        for sub in itertools.combinations(edges, r):
            nodes = {x for e in sub for x in e}
            if sub and g.depot not in nodes:
                continue
            p = list(range(g.n))
            for u, v in sub:
                p[find(p, u)] = find(p, v)
            if sub and len({find(p, x) for x in nodes}) != 1:
                continue
            out.append(sub)
    return out


def test_strong_prune_is_the_best_rooted_subtree():
    rng = np.random.default_rng(11)
    checked = 0
    for i in range(80):
        inst = random_instance(rng, n=int(rng.integers(4, 8)), extra=0, t=int(rng.integers(2, 6)), wmax=5, pmax=7)
        g = A.Graph(inst)
        tree = A.norm_edges((u, v) for u, v, _w in inst.edges)          # kein Extra-Kanten: der Graph selbst ist der Baum
        pruned = A.strong_prune(g, tree)
        best = max(A.net_worth(g, s) for s in _rooted_subtrees(g, tree))
        assert A.net_worth(g, pruned) == pytest.approx(best, abs=EPS), i
        assert A.objective(g, pruned) <= A.objective(g, tree) + EPS
        assert A.strong_prune(g, pruned) == pruned
        checked += 1
    assert checked == 80


def test_strong_prune_of_zero_prizes_is_the_depot_alone():
    inst = S.generate(side=4, t=4, level=0.0, seed=1)
    g = A.Graph(inst)
    assert A.strong_prune(g, A.steiner_all(g).edges) == []


# --- 4. Goemans-Williamson ------------------------------------------------------------------------------------------------------------------------


def test_dual_is_a_lower_bound_and_the_guarantee_holds_on_random_graphs():
    rng = np.random.default_rng(3)
    worst = 0.0
    for i in range(320):
        inst = random_instance(rng, n=int(rng.integers(3, 8)), extra=int(rng.integers(0, 5)), t=int(rng.integers(1, 6)), pmax=int(rng.choice([2, 6, 40])), integer=bool(i % 2))
        g = A.Graph(inst)
        res, sols = A.gw_solutions(g)
        opt = A.pcst_exact(g).value
        assert res.dual <= opt + EPS, i
        n = inst.n
        for name in ("gw", "gws"):
            assert sols[name].objective <= (2 - 1 / (n - 1)) * res.dual + EPS, (i, name)
            assert sols[name].objective <= (2 - 1 / (n - 1)) * opt + EPS, (i, name)
        if res.dual > 1e-9:
            worst = max(worst, sols["gw"].objective / res.dual)
    assert worst > 1.0                                              # Kontrolle: die Schranke wird nicht trivial erfüllt


def test_dual_is_a_lower_bound_and_the_guarantee_holds_on_city_instances():
    for inst in _instances(30, side=4):
        g = A.Graph(inst)
        res, sols = A.gw_solutions(g)
        opt = A.pcst_exact(g).value if inst.t <= 8 else None
        if opt is not None:
            assert res.dual <= opt + EPS
            for name in ("gw", "gws"):
                assert sols[name].objective <= (2 - 1 / (inst.n - 1)) * opt + EPS


def test_gw_bookkeeping():
    for inst in _instances(20):
        g = A.Graph(inst)
        res = A.gw_primal_dual(g)
        times = [e["time"] for e in res.events]
        assert times == sorted(times)
        merges = [e for e in res.events if e["kind"] == "merge"]
        comps = len(set(res.final_comp))
        assert len(merges) == g.n - comps == len(res.forest)
        assert res.dual == pytest.approx(_integral(g, res), abs=1e-7)
        for e in res.events:
            assert e["active"][g.depot] == 0
            assert all(x >= -1e-9 for x in e["d"])
        assert res.events[-1]["n_active"] == 0
        assert all(res.final_comp[v] == res.final_comp[g.depot] or res.marks[v] != -1 for v in range(g.n))


def _integral(g, res):
    """Dual = Summe über die Zeitabschnitte (Zahl aktiver Komponenten davor x Länge), aus den Ereignissen neu berechnet."""
    active = sum(1 for v in g.customers if g.prizes[v] > A.EPS)
    total, last = 0.0, 0.0
    for e in res.events:
        total += active * (e["time"] - last)
        last, active = e["time"], e["n_active"]
    return total


def test_gw_special_cases():
    inst = S.generate(side=4, t=4, level=0.0, seed=2)
    g = A.Graph(inst)
    res, sols = A.gw_solutions(g)
    assert res.events == [] and res.dual == 0.0 and all(s.edges == [] for s in sols.values())
    far = S.Instance(np.zeros((3, 2)), ((0, 1, 10.0), (1, 2, 10.0)), 0, (2,), (0.0, 0.0, 3.0), 0)
    g = A.Graph(far)
    res, sols = A.gw_solutions(g)
    assert all(s.edges == [] for s in sols.values()) and res.dual == pytest.approx(3.0)


def test_textbook_by_hand():
    g = A.Graph(S.textbook_instance(3.0))
    res, sols = A.gw_solutions(g)
    assert res.dual == pytest.approx(5.6)
    for name in ("gwr", "gw", "gws"):
        assert sols[name].objective == pytest.approx(5.6) and sols[name].connected == [4, 5] and sols[name].cost == pytest.approx(5.0)
    assert A.pcst_exact(g).value == pytest.approx(5.6)
    assert A.steiner_all(g).objective == pytest.approx(6.0)
    assert A.connect_none(g).objective == pytest.approx(6.6)
    assert [(e["kind"], e["time"]) for e in res.events][:1] == [("dead", pytest.approx(0.6))]


# --- 5. Schrankenkette ----------------------------------------------------------------------------------------------------------------------------


def test_bound_chain():
    for inst in _instances(30, side=4):
        g = A.Graph(inst)
        if inst.t > 9:
            continue
        res, sols = _all_solutions(g)
        opt = sols["exact"].objective
        assert res.dual <= opt + EPS
        for name, s in sols.items():
            assert s.objective >= opt - EPS, name
        assert sols["gws"].objective <= sols["gw"].objective + EPS
        assert sols["gws"].objective <= sols["gwr"].objective + EPS
        assert sols["stp"].objective <= sols["all"].objective + EPS
        assert sols["ls"].objective <= min(sols["gws"].objective, sols["stp"].objective) + EPS
        assert opt <= min(sols["all"].objective, sols["none"].objective) + EPS


# --- 6. Lokalsuche --------------------------------------------------------------------------------------------------------------------------------


def test_local_search_is_a_local_optimum():
    for inst in _instances(14, side=4):
        g = A.Graph(inst)
        start = A.gw_solutions(g)[1]["gwr"]
        ls = A.local_search(g, start.edges)
        assert ls.objective <= start.objective + EPS
        X = {x for e in ls.edges for x in e} | {g.depot}
        for v in range(g.n):
            Y = X | {v} if v not in X else (X - {v} if v != g.depot else None)
            if Y is None:
                continue
            tree = A.induced_pcst_tree(g, Y)
            if tree is not None and (v in X or any(x in X for x in g.adj[v])):
                assert A.objective(g, tree) >= ls.objective - EPS


# --- 7. Determinismus -----------------------------------------------------------------------------------------------------------------------------


def test_determinism_and_instances_are_connected():
    inst = S.generate(side=6, t=8, blocked=0.3, seed=9)
    g1, g2 = A.Graph(inst), A.Graph(S.generate(side=6, t=8, blocked=0.3, seed=9))
    a, b = A.gw_solutions(g1), A.gw_solutions(g2)
    assert a[0].forest == b[0].forest and a[0].dual == b[0].dual
    assert all(a[1][k].edges == b[1][k].edges for k in a[1])
    assert np.isfinite(g1.dist).all()


# --- 8. Nestung: Gegenbeispiel (Brute-Force) ------------------------------------------------------------------------------------------------------


def _optimal_trees(inst):
    """(Zielwert, Kundenmengen aller optimalen Bäume) über ALLE Baumkantenmengen durch das Depot (Brute-Force)."""
    edges = [(u, v) for u, v, _w in inst.edges]
    best, sets = math.inf, []
    for r in range(len(edges) + 1):
        for sub in itertools.combinations(edges, r):
            p = list(range(inst.n))
            ok = True
            for u, v in sub:
                a, b = find(p, u), find(p, v)
                if a == b:
                    ok = False
                    break
                p[a] = b
            nodes = {x for e in sub for x in e}
            if not ok or (sub and inst.depot not in nodes) or (sub and len({find(p, x) for x in nodes}) != 1):
                continue
            val = objective_of(inst, sub)
            conn = frozenset(c for c in inst.customers if c in nodes)
            if val < best - 1e-9:
                best, sets = val, [conn]
            elif abs(val - best) <= 1e-9:
                sets.append(conn)
    return best, set(sets)


def test_nesting_counterexample_by_brute_force():
    from fixtures import NEST_SCALES, nesting_instance
    lo, hi = (nesting_instance(s) for s in NEST_SCALES)
    v_lo, sets_lo = _optimal_trees(lo)
    v_hi, sets_hi = _optimal_trees(hi)
    assert v_hi >= v_lo - 1e-9
    assert all(not a <= b for a in sets_lo for b in sets_hi), (sets_lo, sets_hi)
    for inst, v in ((lo, v_lo), (hi, v_hi)):
        assert A.pcst_exact(A.Graph(inst)).value == pytest.approx(v, abs=EPS)
    collected = lambda inst, sets: max(sum(inst.prizes[c] for c in s) for s in sets)
    assert collected(hi, sets_hi) >= collected(lo, sets_lo) - 1e-9          # der gesammelte Erlös fällt trotzdem nicht
