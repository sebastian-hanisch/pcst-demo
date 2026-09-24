"""Auswertung: Analyse, Referenz und Lücken, Netto-Lücke, run_config, Sweeps, Erlösniveau-Kurve, Nestung, Qualitäts- und Zeitexperiment."""

from dataclasses import replace

import numpy as np
import pytest

import pcst_algorithm as A
import pcst_constants as C
import pcst_evaluation as ev

FAST = ev.Settings(side=5, t=5, seed=3)


def test_analyse_has_all_methods_and_the_exact_one_when_offered():
    a = ev.analyse(FAST)
    assert set(a.sols) == {"gwr", "gw", "gws", "all", "stp", "none", "ls", "exact"} and a.exact_offered and a.proved
    assert a.states == (2 ** 5 - 1) * 25 and a.tables is not None
    assert a.ref == pytest.approx(a.sols["exact"].objective) and a.gap("exact") == 0.0 and all(a.gap(k) >= 0.0 for k in a.sols)


def test_exact_is_offered_up_to_n_exact_only():
    on = ev.analyse(replace(FAST, side=6, t=C.N_EXACT))
    off = ev.analyse(replace(FAST, side=6, t=C.N_EXACT + 1))
    assert on.exact_offered and "exact" in on.sols
    assert not off.exact_offered and "exact" not in off.sols and off.ref == pytest.approx(off.best.objective) and off.states == 0


def test_best_is_the_lowest_objective_and_reference_semantics():
    for seed in range(6):
        a = ev.analyse(replace(FAST, seed=seed, level=2.5))
        assert a.best.objective == pytest.approx(min(s.objective for s in a.sols.values()))
        assert a.best.objective == pytest.approx(a.ref)                  # mit Exakt ist der beste Fund das Optimum
        base = min(a.sols["all"].objective, a.sols["none"].objective)
        expected = 0.0 if a.ref >= base - 1e-9 else 100.0 * (1.0 - a.ref / base)
        assert a.select_gain_pct == pytest.approx(expected)
        assert 0.0 <= a.select_gain_pct <= 100.0


def test_net_gap_definition():
    a = ev.analyse(replace(FAST, level=3.0))
    assert a.ref_net > 0
    for k in ("gw", "gws", "stp", "ls", "all", "none"):
        expected = max(0.0, 100.0 * (a.ref_net - a.sols[k].net) / a.ref_net)
        assert a.net_gap(k) == pytest.approx(expected)
    assert a.net_gap("exact") == 0.0
    low = ev.analyse(replace(FAST, level=0.5))
    assert low.ref_net <= 1e-9 and low.net_gap("gws") is None                # kein positiver Referenz-Netto-Erlös: keine Netto-Lücke


def test_dual_ratio_is_at_most_one_and_positive():
    for seed in range(8):
        for lv in (1.5, 3.0, 8.0):
            a = ev.analyse(replace(FAST, seed=seed, level=lv))
            assert 0.0 < a.dual_ratio <= 1.0 + 1e-9, (seed, lv)


def test_run_config_keys_and_ordering_of_statistics():
    out = ev.run_config(FAST, level=2.5)
    assert out["n_runs"] == 5 and out["offered_share"] == 100.0 and out["guarantee_violations"] == 0
    for key in ("ref", "dual_ratio", "select_gain_pct", "connected", "connected_share", "gap_gw", "gap_gws", "gap_stp", "gap_ls", "net_gap_gws"):
        assert f"{key}_lo" in out and f"{key}_hi" in out
        lo, med, hi = out[f"{key}_lo"], out[key], out[f"{key}_hi"]
        if not np.isnan(med):
            assert lo <= med + 1e-9 <= hi + 2e-9
    assert all(0.0 <= out[f"{nm}_optimal_share"] <= 100.0 for nm in ("gw", "gws", "stp", "ls"))


def test_sweeps_cover_every_parameter():
    small = replace(FAST, side=5, t=4)
    for param, values in (("level", (1.0, 3.0)), ("t", (3, 5)), ("side", (5, 6)), ("blocked", (0.0, 0.2)), ("layout", C.LAYOUTS), ("prize_mode", C.PRIZE_MODES)):
        rows = ev.sweep(param, small, values)
        assert [r["value"] for r in rows] == list(values) and all(r["n_runs"] == 5 for r in rows)
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS) and all(all(v in C.LEVELS for v in ev.SWEEP_VALUES["level"]) for _ in [0])


def test_level_curve_of_a_city_instance():
    base = replace(FAST, t=6)
    rows = ev.level_curve(base, levels=(0.5, 1.5, 2.5, 3.0, 5.0, 20.0))
    nets = [r["opt"]["net"] for r in rows]
    assert nets == sorted(nets)                                              # der Netto-Erlös des Optimums fällt mit dem Niveau nie
    assert rows[0]["opt"]["connected"] == 0 and rows[-1]["opt"]["connected"] == 6
    for r in rows:
        assert r["opt_name"] == "exact" and r["opt"]["objective"] <= min(r[k]["objective"] for k in ("gw", "gws", "stp", "all", "none")) + 1e-9
        assert r["none"]["connected"] == 0 and r["none"]["objective"] == pytest.approx(r["total_prize"])
        assert r["opt"]["net"] == pytest.approx(r["total_prize"] - r["opt"]["objective"])
    # dasselbe wie eine einzelne Analyse bei diesem Niveau
    a = ev.analyse(replace(base, level=2.5))
    row = next(r for r in rows if r["level"] == 2.5)
    assert row["opt"]["objective"] == pytest.approx(a.ref) and row["gws"]["objective"] == pytest.approx(a.sols["gws"].objective) and row["all"]["objective"] == pytest.approx(a.sols["all"].objective)


def test_level_curve_of_the_textbook_by_hand():
    rows = ev.level_curve(ev.Settings(kind="textbook"), levels=(2.0, 2.5, 3.0, 4.0))
    by = {r["level"]: r for r in rows}
    assert by[2.0]["opt"]["objective"] == pytest.approx(4.6) and by[2.0]["opt"]["connected"] == 0
    assert by[2.5]["opt"]["objective"] == pytest.approx(5.6)                # Gleichstand: nichts anschließen kostet dasselbe wie A und B anschließen
    assert by[3.0]["opt"]["objective"] == pytest.approx(5.6) and by[3.0]["opt"]["connected"] == 2
    assert by[4.0]["opt"]["objective"] == pytest.approx(5.6) and by[4.0]["all"]["objective"] == pytest.approx(6.0)


def test_level_curve_without_exact_uses_the_best_found():
    rows = ev.level_curve(ev.Settings(side=6, t=15, seed=1), levels=(1.0, 3.0))
    assert all(r["opt_name"] != "exact" and "exact" not in r for r in rows)


def test_nesting_returns_examples_and_counts():
    n = ev.nesting(ev.Settings(prize_mode="mixed"), seeds=[12, 13])
    assert n["n_runs"] == 2 and n["violated"] == 1 and n["prize_drops"] == 0
    assert n["examples"] == [{"seed": 12, "from": 3.0, "to": 3.25}]
    none = ev.nesting(ev.Settings(prize_mode="equal", t=5, side=5), seeds=range(200000, 200010))
    assert none["violated"] == 0 and none["violated_share"] == 0.0
    big = ev.nesting(ev.Settings(t=C.N_EXACT + 1), seeds=[1])
    assert big["n_runs"] == 0


def test_quality_small():
    q = ev.quality(FAST, seeds=range(200000, 200008))
    assert q["n_runs"] == 8 and q["exact_used"] and q["guarantee_violations"] == 0
    assert 0.0 <= q["gws_optimal"] <= 100.0 and q["gws_gap_max"] >= q["gws_gap_mean"] >= 0.0 and 0.0 < q["dual_ratio_min"] <= q["dual_ratio_mean"] <= 1.0
    assert q["stp_beats_gws"] + q["gws_beats_stp"] <= 100.0 + 1e-9


def test_exact_time_curve_states():
    rows = ev.exact_time_curve(replace(FAST, side=6), ts=[4, 5, 6])
    assert [r["t"] for r in rows] == [4, 5, 6] and all(r["states"] == (2 ** r["t"] - 1) * 36 and r["seconds"] > 0 for r in rows)


def test_instance_of_clamps_customers_to_the_grid_and_supports_the_textbook():
    inst = ev.instance_of(ev.Settings(side=5, t=30))
    assert inst.t == 24
    assert ev.instance_of(ev.Settings(kind="textbook", level=4.0)).prizes[4] == 4.0


def test_exact_tables_best_matches_pcst_exact_for_other_prizes():
    a = ev.analyse(FAST)
    tab = a.tables
    S_, val = tab.best(a.g.prizes * 0.5)
    g2 = a.g.with_prizes(a.g.prizes * 0.5)
    assert A.pcst_exact(g2).value == pytest.approx(val)
    assert A.objective(g2, tab.edges_of(S_)) <= val + 1e-9
