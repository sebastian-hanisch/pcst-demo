"""Instanz: Kopie treu zur Steiner-Baum-Demo, Aufbau, Erlöse (gleich / gemischt / Niveau), Lehrbuchbeispiel, Fehlerfälle."""

import numpy as np
import pytest

import pcst_constants as C
import pcst_scenario as S
from pcst_unionfind import UnionFind


def test_city_matches_the_steiner_tree_demo_for_the_same_terminals():
    """Die Kopie ist treu: mit t + 1 Terminals der Steiner-Baum-Demo (Depot = kleinstes Terminal) entstehen derselbe Plan, dieselben Straßen und dieselben Sperrungen (Zahlen aus steiner-tree-demo)."""
    i = S.generate(8, 6, 0.1, "uniform", seed=35)
    assert (i.depot,) + i.customers == (15, 18, 28, 35, 50, 59, 62)
    assert i.m == 101 and len(i.blocked_edges) == 11
    assert i.edges[0][2] == pytest.approx(8.660044, abs=1e-6) and sum(w for _u, _v, w in i.edges) == pytest.approx(1014.5189, abs=1e-3)
    j = S.generate(6, 8, 0.3, "clusters", seed=12)
    assert (j.depot,) + j.customers == (13, 18, 19, 20, 24, 25, 26, 30, 32)
    assert j.m == 42 and len(j.blocked_edges) == 18 and sum(w for _u, _v, w in j.edges) == pytest.approx(428.3736, abs=1e-3)


@pytest.mark.parametrize("layout", C.LAYOUTS)
@pytest.mark.parametrize("blocked", C.BLOCKED_OPTIONS)
def test_instance_structure(layout, blocked):
    inst = S.generate(side=7, t=9, blocked=blocked, layout=layout, seed=3)
    assert inst.n == 49 and inst.t == 9 and inst.depot == min(inst.terminals) and inst.depot not in inst.customers
    assert list(inst.customers) == sorted(set(inst.customers)) and len(inst.terminals) == 10
    assert all(u < v for u, v, _w in inst.edges) and list(inst.edges) == sorted(inst.edges)
    assert all(w > 0 for _u, _v, w in inst.edges)
    uf = UnionFind(inst.n, "full")
    for u, v, _w in inst.edges:
        uf.union(u, v)
    assert uf.components == 1
    assert len(inst.blocked_edges) + inst.m == 2 * 7 * 6
    assert all(inst.prizes[v] == 0.0 for v in range(inst.n) if v not in inst.customers)
    assert all(inst.prizes[c] > 0 for c in inst.customers)


def test_determinism_and_seed_dependence():
    a, b, c = S.generate(seed=5), S.generate(seed=5), S.generate(seed=6)
    assert a.edges == b.edges and a.customers == b.customers and a.prizes == b.prizes
    assert a.customers != c.customers or a.edges != c.edges


def test_equal_prizes_scale_with_the_level():
    for level in (0.5, 1.0, 2.5, 20.0):
        inst = S.generate(side=6, t=7, level=level, seed=2)
        assert {inst.prizes[c] for c in inst.customers} == {level * C.SPACING}
        assert inst.total_prize == pytest.approx(7 * level * C.SPACING)
    lo, hi = S.generate(side=6, t=7, level=1.0, prize_mode="mixed", seed=2), S.generate(side=6, t=7, level=2.0, prize_mode="mixed", seed=2)
    assert all(hi.prizes[v] == pytest.approx(2 * lo.prizes[v]) for v in range(lo.n))


def test_mixed_prizes_stay_in_the_band_and_are_deterministic():
    inst = S.generate(side=7, t=20, level=3.0, prize_mode="mixed", seed=4)
    vals = [inst.prizes[c] for c in inst.customers]
    assert all(0.5 * 3.0 * C.SPACING <= v <= 1.5 * 3.0 * C.SPACING for v in vals) and len(set(vals)) == 20
    assert vals == [S.generate(side=7, t=20, level=3.0, prize_mode="mixed", seed=4).prizes[c] for c in inst.customers]
    assert vals != [S.generate(side=7, t=20, level=3.0, prize_mode="mixed", seed=5).prizes[c] for c in inst.customers]


def test_level_zero_gives_zero_prizes():
    inst = S.generate(level=0.0)
    assert inst.total_prize == 0.0


def test_textbook_instance_by_hand():
    inst = S.textbook_instance()
    assert inst.n == 7 and inst.m == 7 and inst.depot == 0 and inst.customers == (4, 5, 6) and inst.kind == "textbook"
    assert inst.prizes == (0.0, 0.0, 0.0, 0.0, 3.0, 3.0, C.TEXTBOOK_SMALL_PRIZE)
    assert all(w == 1.0 for u, v, w in inst.edges if (u, v) != (4, 5)) and dict(((u, v), w) for u, v, w in inst.edges)[(4, 5)] == 2.0
    assert S.textbook_instance(2.0).prizes[4] == 2.0 and S.textbook_instance(2.0).prizes[6] == C.TEXTBOOK_SMALL_PRIZE


def test_errors():
    with pytest.raises(ValueError):
        S.generate(layout="ring")
    with pytest.raises(ValueError):
        S.generate(prize_mode="random")
    with pytest.raises(ValueError):
        S.generate(t=0)
    with pytest.raises(ValueError):
        S.generate(side=3, t=9)
    with pytest.raises(ValueError):
        S.generate(level=-1.0)


def test_levels_and_defaults_are_consistent():
    assert C.DEFAULT_LEVEL in C.LEVELS and 3.0 in C.LEVELS and list(C.LEVELS) == sorted(C.LEVELS)
    assert C.DEFAULT_BLOCKED in C.BLOCKED_OPTIONS and C.T_MIN <= C.DEFAULT_T <= C.T_MAX and C.SIDE_MIN <= C.DEFAULT_SIDE <= C.SIDE_MAX
    assert np.isclose(C.LEVELS[0], 0.5) and np.isclose(C.LEVELS[-1], 20.0)
