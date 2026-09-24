"""Presets: gültige Werte, Bänder (Median über die 5 festen Instanzen), und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen."""

import pytest

import pcst_constants as C
import pcst_evaluation as ev
from pcst_presets import PRESET_KEYS, SETTING_SPECS


def _settings(p):
    return ev.Settings(p["kind"], p["side"], p["t"], p["blocked"], p["layout"], p["prize_mode"], p["level"], p["seed"])


def _analysis(name):
    return ev.analyse(_settings(C.PRESETS[name]))


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 9
    for name, p in C.PRESETS.items():
        assert set(p) == set(PRESET_KEYS), name
        for key, state_key in PRESET_KEYS.items():
            spec = SETTING_SPECS[state_key]
            assert spec.caster(p[key]) == p[key], (name, key)
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip(), name
        assert p["t"] + 1 <= p["side"] ** 2


@pytest.mark.parametrize("name", [n for n in C.PRESET_EXPECTED_BANDS])
def test_preset_bands_over_the_five_fixed_instances(name):
    p = C.PRESETS[name]
    metric, lo, hi = C.PRESET_EXPECTED_BANDS[name]
    assert lo <= ev.run_config(_settings(p))[metric] <= hi


def _has(name, *values):
    text = C.PRESET_HELP[name]
    for v in values:
        assert v in text, (name, v)


def test_help_standard():
    a = _analysis("Standardfall (Voreinstellung)")
    assert (a.inst.n, a.inst.m, len(a.inst.blocked_edges), a.t, a.inst.total_prize) == (64, 101, 11, 8, 200.0)
    assert len(a.sols["exact"].connected) == 5
    _has("Standardfall (Voreinstellung)", f"{a.sols['exact'].cost:.2f}", f"{a.sols['exact'].net:.2f}", f"{a.ref:.2f}", f"{a.sols['all'].objective:.2f}", f"+{a.gap('all'):.2f} %", f"+{a.gap('none'):.2f} %",
         f"{a.select_gain_pct:.2f} %", f"+{a.gap('gws'):.2f} %", f"+{a.gap('gw'):.2f} %")
    assert a.gap("stp") == 0.0 and a.gap("ls") == 0.0 and len(a.sols["gw"].connected) == 8


def test_help_textbook():
    a = _analysis("Lehrbuchbeispiel (Gabel)")
    assert a.ref == pytest.approx(5.6) and a.sols["exact"].cost == pytest.approx(5.0) and a.sols["exact"].net == pytest.approx(1.0) and a.dual == pytest.approx(5.6)
    assert a.sols["all"].objective == pytest.approx(6.0) and a.sols["none"].objective == pytest.approx(6.6)
    other = ev.analyse(ev.Settings(kind="textbook", level=2.5))
    assert other.sols["none"].objective == pytest.approx(5.6) and other.ref == pytest.approx(5.6)
    _has("Lehrbuchbeispiel (Gabel)", "Zielwert 5.6", "Kosten 5", "6.6", "5.6")


def test_help_low_level():
    a = _analysis("Niedriges Erlösniveau")
    assert len(a.sols["exact"].connected) == 3 and len(a.sols["gw"].connected) == 7 and len(a.sols["gws"].connected) == 2
    _has("Niedriges Erlösniveau", f"{a.ref:.2f}", f"{a.select_gain_pct:.2f} %", f"+{a.gap('all'):.2f} %", f"+{a.gap('gw'):.2f} %", f"+{a.gap('gws'):.2f} %")
    assert a.gap("stp") == 0.0 and a.gap("ls") == 0.0


def test_help_high_level():
    a = _analysis("Hohes Erlösniveau")
    assert len(a.sols["exact"].connected) == 8 and a.gap("all") == 0.0 and a.gap("stp") == 0.0 and a.gap("ls") == 0.0
    assert a.sols["gw"].objective == pytest.approx(a.sols["gws"].objective) and a.sols["gw"].edges == a.sols["gws"].edges
    _has("Hohes Erlösniveau", f"{a.ref:.2f}", f"+{a.gap('none'):.2f} %", f"+{a.gap('gws'):.2f} %", f"{a.dual_ratio * 100:.0f} %")


def test_help_clusters():
    a = _analysis("Gruppierte Kunden")
    assert a.t == 9 and len(a.sols["exact"].connected) == 5 and a.gap("gws") == 0.0 and a.gap("stp") == 0.0 and a.gap("ls") == 0.0
    _has("Gruppierte Kunden", f"{a.ref:.2f}", f"+{a.gap('all'):.2f} %", f"+{a.gap('none'):.2f} %", f"{a.select_gain_pct:.2f} %", f"+{a.gap('gw'):.2f} %")


def test_help_gw_far_above_the_optimum():
    a = _analysis("GW weit über dem Optimum")
    assert a.gap("gws") > 10.0 and a.gap("stp") < 1.0
    _has("GW weit über dem Optimum", f"{a.sols['gws'].objective:.2f}", f"+{a.gap('gws'):.2f} %", f"{a.ref:.2f}", f"{a.sols['gw'].objective:.2f}", f"+{a.gap('gw'):.2f} %", f"{a.sols['exact'].net:.2f}",
         f"{a.sols['gws'].net:.2f}", f"−{a.net_gap('gws'):.1f} %", f"{a.sols['stp'].objective:.2f}", f"+{a.gap('stp'):.2f} %", f"{a.dual:.2f}", f"{a.dual_ratio * 100:.0f} %")


def test_help_build_then_prune_loses_to_gw():
    a = _analysis("Erst Steiner verliert gegen GW")
    assert a.gap("gw") == 0.0 and a.gap("gws") == 0.0 and a.gap("ls") == 0.0 and a.gap("stp") > 5.0
    _has("Erst Steiner verliert gegen GW", f"{a.sols['stp'].objective:.2f}", f"+{a.gap('stp'):.2f} %", f"{a.ref:.2f}", f"{a.sols['exact'].net:.2f}", f"{a.sols['stp'].net:.2f}", f"−{a.net_gap('stp'):.1f} %")


def test_help_local_search_stuck():
    a = _analysis("Lokalsuche bleibt hängen")
    assert a.gap("ls") > 5.0 and a.sols["ls"].objective == pytest.approx(a.sols["stp"].objective) and len(a.sols["exact"].connected) == 7
    _has("Lokalsuche bleibt hängen", f"{a.sols['ls'].objective:.2f}", f"+{a.gap('ls'):.2f} %", f"{a.ref:.2f}", f"+{a.gap('gws'):.2f} %")


def test_nesting_preset():
    p = C.PRESETS["Nestung verletzt"]
    low, high = (ev.analyse(ev.Settings(p["kind"], p["side"], p["t"], p["blocked"], p["layout"], p["prize_mode"], lv, p["seed"])).sols["exact"].connected for lv in (3.0, p["level"]))
    assert len(low) == 6 and len(high) == 7 and 58 in low and 58 not in high and {12, 22} <= set(high) and not ({12, 22} & set(low))
    assert p["level"] == 3.25 and 3.0 in C.LEVELS
    _has("Nestung verletzt", "58", "12 und 22", "3.25")


def test_presets_cover_the_transition():
    """Zwischen den Presets liegt der Übergang: niedriges Niveau schließt wenig, hohes alle an."""
    low = ev.run_config(_settings(C.PRESETS["Niedriges Erlösniveau"]))
    high = ev.run_config(_settings(C.PRESETS["Hohes Erlösniveau"]))
    assert low["connected_share"] < 15.0 and high["connected_share"] == 100.0
