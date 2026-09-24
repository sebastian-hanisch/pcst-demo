"""Jede Zahl in README und App-Texten (Grenzen-Tabelle, Hypothesen) über die echten Auswertungsfunktionen `ev.*` - nie über ein Ad-hoc-Skript. Bänder mit Sicherheitsabstand; ganzzahlige Aussagen exakt.
Standard: Plan 8 x 8, 8 Kunden, 10 % gesperrt, gleiche Erlöse; 50 Instanzen Seeds 200000-200049 (`quality`) bzw. 5 feste Instanzen Seeds 100000-100004 (`run_config`)."""

from dataclasses import replace
from functools import lru_cache

import pytest

import pcst_constants as C
import pcst_evaluation as ev

B = ev.Settings()


@lru_cache(maxsize=None)
def Q(**kw):
    return ev.quality(replace(B, **kw))


@lru_cache(maxsize=None)
def R(**kw):
    return ev.run_config(B, **kw)


def within(x, lo, hi):
    assert lo <= x <= hi, (x, lo, hi)


# --- 1. Wert der Auswahl: am größten mitten im Übergang ---------------------------------------------------------------------------------------------


def test_value_of_the_selection_peaks_in_the_middle():
    lo, mid, hi = Q(level=1.5), Q(level=2.5), Q(level=4.0)
    within(lo["select_share"], 10, 26), within(lo["select_gain_mean"], 0.6, 1.7)
    within(mid["select_share"], 82, 96), within(mid["select_gain_mean"], 5.0, 7.0)
    within(hi["select_share"], 48, 68), within(hi["select_gain_mean"], 1.2, 2.3)
    assert mid["select_gain_mean"] > lo["select_gain_mean"] and mid["select_gain_mean"] > hi["select_gain_mean"]
    within(mid["connected_mean"], 5.0, 6.0)
    within(lo["connected_mean"], 0.2, 0.6), within(hi["connected_mean"], 7.5, 8.0)


def test_alle_and_nichts_baselines_at_the_middle_level():
    mid = Q(level=2.5)
    within(mid["all_gap_mean"], 8.0, 10.3), within(mid["none_gap_mean"], 14.5, 18.5)
    within(Q(level=1.5)["all_gap_mean"], 52.0, 68.0), within(Q(level=4.0)["none_gap_mean"], 66.0, 82.0)


def test_transition_over_the_levels_on_the_five_fixed_instances():
    shares = {lv: R(level=lv)["connected_share"] for lv in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)}
    assert shares == {1.5: 0.0, 2.0: 62.5, 2.5: 87.5, 3.0: 100.0, 4.0: 100.0, 6.0: 100.0}


def test_clusters_shift_the_transition_to_lower_levels():
    low, mid = Q(layout="clusters", t=9, level=1.5), Q(layout="clusters", t=9, level=2.5)
    within(low["select_share"], 70, 86), within(low["select_gain_mean"], 4.3, 6.0), within(low["connected_mean"], 5.5, 6.6)
    within(mid["select_share"], 4, 16), within(mid["select_gain_mean"], 0.05, 0.4), within(mid["connected_mean"], 8.4, 9.0)


def test_blockages_make_customers_drop_out():
    open_, shut = Q(blocked=0.0), Q(blocked=0.3)
    within(open_["connected_mean"], 5.8, 6.8), within(shut["connected_mean"], 2.6, 3.5)
    within(shut["select_share"], 50, 70), within(open_["select_share"], 70, 86)


def test_mixed_prizes_behave_like_equal_ones():
    mixed = Q(prize_mode="mixed")
    within(mixed["select_share"], 76, 92), within(mixed["select_gain_mean"], 5.8, 7.8), within(mixed["connected_mean"], 4.4, 5.4)


# --- 2. Goemans-Williamson gegen exakt ---------------------------------------------------------------------------------------------------------------


def test_gw_at_the_middle_level():
    q = Q(level=2.5)
    assert q["exact_used"] and q["guarantee_violations"] == 0
    within(q["gw_optimal"], 0, 4), within(q["gw_gap_mean"], 11.5, 14.0), within(q["gw_gap_max"], 25.0, 32.0)
    within(q["gws_optimal"], 22, 38), within(q["gws_gap_mean"], 2.3, 3.4), within(q["gws_gap_max"], 10.0, 15.0)
    within(q["gws_better_than_gw"], 80, 95)
    within(q["gwr_gap_mean"], 150, 180)


def test_build_then_prune_and_local_search_beat_gw():
    q = Q(level=2.5)
    within(q["stp_optimal"], 48, 64), within(q["stp_gap_mean"], 0.6, 1.1), within(q["stp_gap_max"], 3.5, 5.0)
    within(q["ls_optimal"], 52, 68), within(q["ls_gap_mean"], 0.5, 0.95), within(q["ls_gap_max"], 3.5, 5.0)
    within(q["stp_beats_gws"], 50, 66), within(q["gws_beats_stp"], 3, 13)
    assert q["stp_gap_mean"] < q["gws_gap_mean"] < q["gw_gap_mean"]


def test_the_dual_lower_bound():
    q = Q(level=2.5)
    within(q["dual_ratio_mean"], 0.64, 0.72), within(q["dual_ratio_min"], 0.53, 0.63)
    assert Q(level=1.5)["dual_ratio_mean"] > q["dual_ratio_mean"] > Q(level=4.0)["dual_ratio_mean"] - 0.06
    within(Q(level=1.5)["dual_ratio_mean"], 0.88, 0.97), within(Q(level=4.0)["dual_ratio_mean"], 0.60, 0.68)
    assert all(Q(level=lv)["guarantee_violations"] == 0 for lv in (1.5, 2.5, 4.0))


def test_gw_gap_depends_on_the_level_and_on_the_pruning():
    low, high = Q(level=1.5), Q(level=4.0)
    within(low["gw_gap_mean"], 19.0, 25.0), within(low["gws_gap_mean"], 0.0, 0.01), assert_eq(low["gws_optimal"], 100.0), within(low["gw_optimal"], 26, 42)
    within(high["gw_gap_mean"], 6.0, 7.7), within(high["gws_gap_mean"], 5.4, 6.9), within(high["stp_gap_mean"], 0.6, 1.2)
    within(high["stp_beats_gws"], 78, 94), assert_eq(high["gws_beats_stp"], 0.0)
    within(high["gws_better_than_gw"], 16, 32)


def assert_eq(x, y):
    assert x == y


def test_more_customers_and_fewer_customers():
    small = Q(t=5)
    within(small["gws_optimal"], 76, 92), within(small["connected_mean"], 1.5, 2.3), within(small["select_share"], 56, 72)


# --- 3. Netto-Erlös gegen Zielwert -------------------------------------------------------------------------------------------------------------------


def test_small_objective_gaps_are_large_net_losses():
    q = Q(level=2.5)
    within(q["gws_net_gap_mean"], 22.0, 27.0), within(q["stp_net_gap_mean"], 8.0, 11.5), within(q["gw_net_gap_mean"], 140.0, 190.0), within(q["ls_net_gap_mean"], 7.5, 10.5)
    assert q["gws_net_gap_mean"] > 5 * q["gws_gap_mean"] and q["stp_net_gap_mean"] > 8 * q["stp_gap_mean"] and q["gw_net_gap_mean"] > 10 * q["gw_gap_mean"]


def test_the_five_fixed_instances_over_the_levels():
    lo, mid, mid3 = R(level=1.5), R(level=2.5), R(level=3.0)
    within(lo["gap_gw"], 22.0, 27.0), within(lo["gap_gws"], 0.0, 0.01)
    within(mid["gap_gw"], 10.0, 13.5), within(mid["gap_gws"], 1.0, 2.8), within(mid["select_gain_pct"], 4.0, 6.0), within(mid["dual_ratio"], 0.62, 0.70)
    within(mid3["gap_gw"], 5.0, 7.0), within(mid3["gap_gws"], 3.4, 4.6), within(mid3["net_gap_gws"], 8.0, 11.0)
    assert R(level=4.0)["gap_gw"] == pytest.approx(R(level=4.0)["gap_gws"]) and R(level=6.0)["net_gap_gws"] < mid3["net_gap_gws"]
    assert all(R(level=lv)["guarantee_violations"] == 0 and R(level=lv)["gap_stp"] == 0.0 and R(level=lv)["gap_ls"] == 0.0 for lv in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0))


# --- 4. Nestung --------------------------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def N(**kw):
    return ev.nesting(replace(B, **kw))


def test_nesting_on_the_default_plan_holds_and_fails_on_larger_ones():
    for mode in ("equal", "mixed"):
        n = N(prize_mode=mode)
        assert n["n_runs"] == 50 and n["violated"] == 0 and n["prize_drops"] == 0
    big = N(t=10, side=10)
    assert big["n_runs"] == 50 and big["violated"] == 2 and big["prize_drops"] == 0
    assert {(x["from"], x["to"]) for x in big["examples"]} == {(3.25, 3.5)}


def test_the_collected_prize_never_falls_with_the_level():
    assert all(N(**kw)["prize_drops"] == 0 for kw in ({}, {"prize_mode": "mixed"}, {"layout": "clusters", "t": 9}, {"blocked": 0.3}))


def test_nesting_preset_instance_and_scan():
    n = ev.nesting(replace(B, prize_mode="mixed"), seeds=range(0, 60))
    assert n["n_runs"] == 60 and n["violated"] >= 1 and {"seed": 12, "from": 3.0, "to": 3.25} in n["examples"] and n["prize_drops"] == 0


# --- 5. Exakter Weg ----------------------------------------------------------------------------------------------------------------------------------


def test_exact_state_counts():
    rows = ev.exact_time_curve(replace(B, side=8), ts=[4, 8, 12])
    assert [r["states"] for r in rows] == [(2 ** t - 1) * 64 for t in (4, 8, 12)]
    assert C.N_EXACT == 12 and rows[-1]["seconds"] < 30.0


def test_local_search_rarely_improves_its_start():
    assert Q(level=2.5)["ls_improves"] == 0.0 and Q(level=1.5)["ls_improves"] == 0.0 and Q(level=4.0)["ls_improves"] == 0.0
    within(Q(prize_mode="mixed")["ls_improves"], 2.0, 8.0)
    assert Q(level=2.5)["ls_optimal"] > Q(level=2.5)["stp_optimal"]                # der Vorsprung kommt vom besseren der beiden Starts
