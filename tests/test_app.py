"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt und jedes GW-Ereignis, beide Instanzen und alle Bäume, Randwerte, Würfel-Knopf, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pcst_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=1, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if step != 1:
        at.select_slider(key="pcst_step").set_value(step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label.startswith(label))


def test_default_run_has_no_exception_and_shows_the_four_metrics():
    at = _run()
    _ok(at)
    assert {"Netto-Erlös", "Angeschlossen", "GW (stark)", "Erst Steiner"} <= {m.label for m in at.metric}
    assert _metric(at, "Netto-Erlös").value == "34.85" and _metric(at, "Angeschlossen").value == "5 von 8" and _metric(at, "GW (stark)").value == "+1.07 %" and _metric(at, "Erst Steiner").value == "optimal"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["side_slider"], ss["t_slider"], ss["blocked_select"], ss["layout_select"], ss["prize_select"], ss["level_select"], ss["seed_input"], ss["tree_select"]) == (
        p["kind"], p["side"], p["t"], p["blocked"], p["layout"], p["prize_mode"], p["level"], p["seed"], p["tree"])
    assert at.metric and at.get("plotly_chart")


@pytest.mark.parametrize("step", [1, 2, 3])
def test_every_step_runs_for_every_kind(step):
    for kind in C.KINDS:
        for t in (4, 14):
            at = _run(kind_select=kind, t_slider=t, side_slider=6, pcst_step=step)
            _ok(at)
            assert at.get("plotly_chart") and at.session_state["pcst_step"] == step


def test_every_tree_view_runs():
    for tree in C.TREE_OPTIONS:
        at = _run(step=3, tree_select=tree)
        _ok(at)
        assert at.get("plotly_chart") and any("Netto-Erlös" in m.value and "Kunden" in m.value for m in at.markdown)
    at = _run(step=3, tree_select="exact", t_slider=20, side_slider=8)
    _ok(at)
    assert any("nur bis t = 12" in w.value for w in at.warning)


def test_gw_slider_walks_through_all_events_including_the_last_one():
    at = _run(step=2)
    _ok(at)
    smax = int(at.slider(key="pcst_k").max)
    assert smax == 59
    for k in (0, 30, smax):
        at.slider(key="pcst_k").set_value(k).run()
        _ok(at)
        assert any(x.value.startswith(f"**Ereignis {k} von {smax}") for x in at.markdown)
    assert any("Ende des Wachstums" in x.value for x in at.markdown)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["pcst_k"] <= int(at.slider(key="pcst_k").max) == 6


def test_gw_step_at_the_lowest_level_runs():
    at = _run(step=2, level_select=0.5, t_slider=3)
    _ok(at)
    assert at.get("plotly_chart")


@pytest.mark.parametrize("kw", [
    dict(side_slider=C.SIDE_MIN, t_slider=C.T_MIN), dict(side_slider=C.SIDE_MAX, t_slider=C.T_MAX), dict(side_slider=C.SIDE_MIN, t_slider=C.T_MAX), dict(blocked_select=C.BLOCKED_OPTIONS[-1]),
    dict(layout_select="clusters", t_slider=13), dict(layout_select="clusters", t_slider=C.T_MAX, side_slider=C.SIDE_MAX, blocked_select=0.4), dict(kind_select="textbook"), dict(t_slider=12, side_slider=10),
    dict(level_select=C.LEVELS[0]), dict(level_select=C.LEVELS[-1]), dict(prize_select="mixed", level_select=4.0),
])
def test_extreme_settings_run(kw):
    for step in (1, 2, 3):
        _ok(_run(step=step, **kw))


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(side="99", t="1", blocked="0.15", layout="ring", prize="nope", level="7.0", tree="nope", kind="nope").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["side_slider"], ss["t_slider"], ss["blocked_select"], ss["layout_select"], ss["prize_select"], ss["level_select"], ss["tree_select"], ss["kind_select"]) == (
        C.SIDE_MAX, C.T_MIN, C.DEFAULT_BLOCKED, "uniform", "equal", C.DEFAULT_LEVEL, C.DEFAULT_TREE, "city")


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="city", side="10", t="9", blocked="0.3", layout="clusters", prize="mixed", level="3.25", seed="7", tree="gws").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["side_slider"], ss["t_slider"], ss["blocked_select"], ss["layout_select"], ss["prize_select"], ss["level_select"], ss["seed_input"], ss["tree_select"]) == (10, 9, 0.3, "clusters", "mixed", 3.25, 7, "gws")


def test_sidebar_shows_only_the_controls_that_matter():
    plain = _run()
    assert any(w.key == "side_widget" for w in plain.slider) and any(w.key == "t_widget" for w in plain.slider) and any(w.key == "blocked_widget" for w in plain.select_slider)
    assert any(r.key == "layout_widget" for r in plain.radio) and any(r.key == "prize_widget" for r in plain.radio) and any(n.key == "seed_widget" for n in plain.number_input)
    assert any(w.key == "level_select" for w in plain.select_slider)
    tb = _run(kind_select="textbook")
    assert not any(w.key == "side_widget" for w in tb.slider) and not any(n.key == "seed_widget" for n in tb.number_input)
    assert not any(r.key in ("layout_widget", "prize_widget") for r in tb.radio)
    assert any(w.key == "level_select" for w in tb.select_slider)               # das Niveau wirkt im Lehrbuchbeispiel (Erlös von A und B)


def test_textbook_level_changes_the_answer():
    for level, expected in ((2.0, "0 von 3"), (3.0, "2 von 3"), (5.0, "2 von 3")):
        at = _run(kind_select="textbook", level_select=level)
        _ok(at)
        assert _metric(at, "Angeschlossen").value == expected, level
    assert _metric(_run(kind_select="textbook", level_select=3.0), "Netto-Erlös").value == "1.00"


def test_changing_the_instance_while_on_step_three_does_not_crash():
    at = _run(step=3, tree_select="exact")
    _ok(at)
    for kw in (dict(kind_select="textbook"), dict(kind_select="city", t_slider=20), dict(t_slider=4, side_slider=5), dict(layout_select="clusters"), dict(level_select=6.0)):
        for k, v in kw.items():
            at.session_state[k] = v
        at.run()
        _ok(at)


def test_large_instance_offers_no_exact_method_but_states_it():
    at = _run(t_slider=20, side_slider=8)
    _ok(at)
    assert any("Kein exaktes Verfahren bei t = 20 > 12" in c.value for c in at.caption)


def test_level_curve_section_is_present_and_states_the_transition():
    at = _run()
    _ok(at)
    assert any("Vom leeren Baum" in s.value for s in at.markdown)
    assert any("ab Niveau" in c.value for c in at.caption)


def test_quality_experiment_runs_on_demand():
    at = _run(t_slider=5, side_slider=6)
    next(b for b in at.button if b.key == "quality_start").click().run()
    _ok(at)
    assert {"Auswahl lohnt", "GW (stark) optimal", "Erst Steiner optimal", "Lokalsuche optimal"} <= {m.label for m in at.metric}


def test_nesting_experiment_runs_on_demand():
    at = _run(t_slider=5, side_slider=6)
    next(b for b in at.button if b.key == "nesting_start").click().run()
    _ok(at)
    assert {"Nestung verletzt", "Erlös fällt"} <= {m.label for m in at.metric}


def test_time_curve_runs_on_demand():
    at = _run(side_slider=5)
    next(b for b in at.button if b.key == "time_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


@pytest.mark.parametrize("param", ["level", "t", "side", "blocked", "layout", "prize_mode"])
@pytest.mark.parametrize("metric", ["gap", "net", "select", "dual"])
def test_sweeps_run_on_demand_for_every_metric(param, metric):
    at = _run(side_slider=5, t_slider=4, sweep_metric=metric)
    at.selectbox(key="sweep_select").set_value(param).run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_the_textbook_has_no_experiments():
    tb = _run(kind_select="textbook")
    assert not any(b.key in ("quality_start", "nesting_start", "time_start", "sweep_start") for b in tb.button)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Goemans, M. X., & Williamson, D. P. (1995)" in m.value and "Johnson, D. S., Minkoff, M., & Phillips, S. (2000)" in m.value and "Ahmadi, A." in m.value for m in at.markdown)
