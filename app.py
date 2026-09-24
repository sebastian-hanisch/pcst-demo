"""Prize-Collecting Steiner-Baum – wer wird angeschlossen? Goemans-Williamson, Lokalsuche, exakt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Spannbaum-Reihe der "Konzepte"-Reihe: bisher musste jeder Kunde angeschlossen werden. Bringt ein Anschluss aber einen Erlös, lohnt sich ein Kunde am Ende einer langen Trasse nicht - die
Entscheidung "wer wird angeschlossen" kommt zur Baumfrage dazu. Gemessen werden der Wert der Auswahl über das Erlösniveau, das Primal-Dual-Verfahren von Goemans und Williamson (1995) mit seiner
Untergrenze, zwei Arten des Beschneidens, "erst Steinerbaum, dann kürzen" und eine Lokalsuche gegen das exakte Optimum (Dreyfus-Wagner) und die Frage, ob die Menge der angeschlossenen Kunden mit dem
Erlösniveau nur wächst.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import pandas as pd
import streamlit as st

import pcst_constants as C
from pcst_evaluation import SWEEP_LABELS, SWEEP_TICKS, Settings, analyse, exact_time_curve, level_curve, nesting, quality, sweep
from pcst_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    store_from_widget,
    sync_query_params,
)
from pcst_visualization import (
    METHOD_LABELS,
    build_gw_step,
    build_instance,
    build_level_curve,
    build_net_bars,
    build_sweep,
    build_time_curve,
    build_tree,
)

st.set_page_config(page_title="Prize-Collecting Steiner-Baum – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _curve(settings):
    return level_curve(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _quality(base):
    return quality(base)


@st.cache_data(show_spinner=False)
def _nesting(base):
    return nesting(base)


@st.cache_data(show_spinner=False)
def _time_curve(base):
    return exact_time_curve(base)


def pct(x):
    return f"{x:+.2f} %"


def num(x):
    return f"{x:.2f}"


st.title("💶 Prize-Collecting Steiner-Baum – wer wird angeschlossen?")
st.markdown(
    """
**Neuntes Stück der Spannbaum-Reihe.** Bisher musste **jeder Kunde** angeschlossen werden. Bringt ein Anschluss aber einen **Erlös**, lohnt sich ein Kunde am Ende einer langen Trasse vielleicht nicht - und zwei
Kunden, die einzeln zu weit weg sind, lohnen sich zusammen, weil sie sich einen Stamm teilen. Gegeben ist ein Stadtplan (Kreuzungen und Straßen mit Länge), ein **Depot**, das im Baum liegen muss, und **Kunden mit Erlös**;
gesucht ist der Baum, der die Summe aus **Trassenkosten und den Erlösen der nicht angeschlossenen Kunden** minimiert (der **Zielwert**). Das **Prize-Collecting-Steiner-Problem** ist **NP-schwer**; ohne Erlöse ist die
Antwort der leere Baum, mit sehr hohen Erlösen der Steinerbaum über alle Kunden aus der Vorgänger-Demo.

Hier wird gemessen, **was die Auswahl bringt**, wie gut das **Primal-Dual-Verfahren von Goemans und Williamson** (mit seiner **Untergrenze**) abschneidet, was **starkes Beschneiden** ändert, ob **"erst Steinerbaum, dann kürzen"**
oder eine **Lokalsuche** besser ist, und ob die Menge der angeschlossenen Kunden mit dem Erlösniveau **nur wächst**.
"""
)
st.caption(
    "Setzt auf [steiner-tree-demo](https://github.com/sebastian-hanisch/steiner-tree-demo) auf (Stadtplan, Steinerpunkte, Dreyfus-Wagner, KMB und Takahashi-Matsuyama als Bausteine für \"alle anschließen\"). "
    "Geplante Nachfolger (nicht gebaut): Sensitivität und dynamischer MST, zufällige Spannbäume."
)

with st.expander("So funktionieren die Verfahren", expanded=True):
    st.markdown(
        """
1. **Zielwert und Netto-Erlös:** Zielwert = Trassenkosten + Erlöse der nicht angeschlossenen Kunden. **Netto-Erlös** = Erlöse der angeschlossenen Kunden - Trassenkosten = Gesamterlös - Zielwert. Im Optimum ist beides dasselbe; für
   eine Näherung nicht (siehe unten).
2. **Kernsatz:** jede Lösung ist ein Baum auf einer Knotenmenge X mit Depot; das Optimum ist der kürzeste Spannbaum des von X induzierten Teilplans plus die Erlöse der Kunden außerhalb von X, über alle X.
3. **Goemans-Williamson (GW):** jeder Kunde ist eine Komponente mit einem Erlös als Vorrat. Alle Komponenten ohne Depot **wachsen** (ein Kreis um jeden ihrer Knoten, der **Moat**) und verbrauchen dabei ihren Vorrat; berühren
   sich zwei Moats, wird die Straße dazwischen **fest** und die Komponenten verschmelzen; ist der Vorrat einer Komponente aufgebraucht, **stirbt** sie und wächst nicht mehr. Erreicht eine Komponente das Depot, ist sie fertig.
   Die Summe aller Moat-Radien über die Zeit ist eine **Untergrenze** des Optimums. Danach wird **beschnitten**.
4. **Beschneiden:** *GW-Beschneiden* entfernt so viel wie möglich, ohne eine tote Menge halb anzuschließen; *starkes Beschneiden* (Johnson, Minkoff und Phillips 2000) schneidet den festen Baum **optimal** zurecht: ein Teilbaum
   bleibt nur, wenn er mehr Erlös bringt, als seine Anschlusskante kostet.
5. **Erst Steinerbaum, dann kürzen:** der beste Steinerbaum über alle Kunden (Kou-Markowsky-Berman, Takahashi-Matsuyama und Lokalsuche aus der Vorgänger-Demo), danach starkes Beschneiden.
6. **Lokalsuche:** Knoten einfügen oder entfernen (Kunde oder Steinerpunkt), bewertet mit dem Zielwert des Spannbaums des Teilplans. **Exakt (Dreyfus-Wagner):** dynamische Programmierung über alle Teilmengen der Kunden, Aufwand
   etwa 3^t x Kreuzungen - deshalb nur bis 12 Kunden.
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:3], preset_names[3:6], preset_names[6:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.radio("Instanz", options=list(C.KINDS), format_func=lambda v: C.KIND_LABELS[v], key="kind_select",
                    help="Stadtplan: gestörtes Gitter mit gesperrten Straßen. Lehrbuchbeispiel: eine Gabel mit Ausläufer auf 7 Kreuzungen, von Hand nachzurechnen.")
    if kind != "textbook":
        side = st.slider("Gittergröße (Kreuzungen je Seite)", *bounds("side_slider"), value=int(ss["side_slider"]), key="side_widget", on_change=store_from_widget, args=("side_slider",),
                         help="Der Plan hat Seite x Seite Kreuzungen; je mehr Kreuzungen, desto mehr mögliche Steinerpunkte.")
        t = st.slider("Kunden t", *bounds("t_slider"), value=int(ss["t_slider"]), key="t_widget", on_change=store_from_widget, args=("t_slider",),
                      help=f"Anzahl der Kunden (das Depot kommt dazu). Das exakte Optimum wird bis t = {C.N_EXACT} angeboten; darüber vergleicht die Demo die Verfahren untereinander.")
        blocked = st.select_slider("Gesperrter Anteil der Straßen", options=list(C.BLOCKED_OPTIONS), value=float(ss["blocked_select"]), key="blocked_widget", on_change=store_from_widget, args=("blocked_select",),
                                   format_func=lambda v: f"{v:.0%}", help="Sperrungen machen direkte Wege umständlicher (der Plan bleibt zusammenhängend).")
        layout = st.radio("Lage der Kunden", options=list(C.LAYOUTS), format_func=lambda v: C.LAYOUT_LABELS[v], key="layout_widget", on_change=store_from_widget, args=("layout_select",),
                          index=list(C.LAYOUTS).index(ss["layout_select"]), help="Gleichverteilt über den Plan oder in 2 bis 3 Gruppen.")
        prize_mode = st.radio("Erlös der Kunden", options=list(C.PRIZE_MODES), format_func=lambda v: C.PRIZE_LABELS[v], key="prize_widget", on_change=store_from_widget, args=("prize_select",),
                              index=list(C.PRIZE_MODES).index(ss["prize_select"]), help="Alle Kunden zahlen gleich viel, oder jeder zahlt das 0.5- bis 1.5-fache des Niveaus.")
    else:
        side, t, blocked, layout, prize_mode = C.DEFAULT_SIDE, C.DEFAULT_T, 0.0, "uniform", "equal"
    level = st.select_slider("Erlösniveau (Straßenlängen je Kunde)", options=list(C.LEVELS), key="level_select", format_func=lambda v: f"{v:g}",
                             help="Was ein Anschluss einbringt, gemessen in Länge einer Nachbarstraße. Im Lehrbuchbeispiel zahlen nur die beiden Endkunden A und B dieses Niveau; Kunde C zahlt fest 0.6." if kind == "textbook" else
                             "Was ein Anschluss einbringt, gemessen in Länge einer Nachbarstraße: bei 0.5 lohnt sich nichts, bei 20 lohnt sich alles.")
    if kind != "textbook":
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        seed = C.DEFAULT_SEED

sync_query_params({"kind_select": kind, "side_slider": int(ss["side_slider"]), "t_slider": int(ss["t_slider"]), "blocked_select": float(ss["blocked_select"]), "layout_select": ss["layout_select"],
                   "prize_select": ss["prize_select"], "level_select": float(level), "seed_input": int(ss["seed_input"]), "tree_select": ss["tree_select"]})

settings = Settings(kind, int(side), int(t), float(blocked), layout, prize_mode, float(level), int(seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
g = a.g
best = a.best

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Der Baum in Aktion")
STEP_LABELS = {1: "1 · Der Plan", 2: "2 · Goemans-Williamson wächst", 3: "3 · Wer wird angeschlossen"}
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="pcst_step", format_func=lambda s: STEP_LABELS[s])

if step == 1:
    st.markdown(f"**{inst.n} Kreuzungen**, **{inst.m} befahrbare Straßen**" + (f", {len(inst.blocked_edges)} gesperrt (rot gepunktet)" if inst.blocked_edges else "") +
                f"; **{inst.t} Kunden** mit zusammen **{num(inst.total_prize)}** Erlös (Größe des Kreises = Erlös, Zahl = Erlös in Straßenlängen) und das Depot ⭐. Alle Kunden anschließen kostet mindestens **{num(a.sols['all'].cost)}** an Trassen "
                f"(bester gefundener Steinerbaum), nichts anschließen kostet **{num(inst.total_prize)}** an entgangenem Erlös.")
    st.plotly_chart(build_instance(inst), width="stretch", key="s1_map")
    st.caption("Blau = Kunde, Kreisgröße = Erlös, grün = Depot; graue Linien = Straßen, rot gepunktet = gesperrte Straßen. Jede andere Kreuzung ist ein möglicher Steinerpunkt.")
elif step == 2:
    res = a.res
    smax = len(res.events)
    if "pcst_k" in ss:
        ss["pcst_k"] = min(max(0, int(ss["pcst_k"])), smax)
    k = st.slider("Ereignis des Wachstums", 0, smax, key="pcst_k", help="0 = Start, jede Komponente mit Erlös wächst; jedes weitere Ereignis ist eine Kante, die fest wird, oder eine Komponente, deren Erlös aufgebraucht ist.") if smax > 0 else 0
    if smax == 0:
        st.markdown("**Kein Ereignis:** kein Kunde hat einen Erlös, es wächst nichts - der Baum ist leer.")
    elif k == 0:
        st.markdown(f"**Ereignis 0 von {smax}:** jeder Kunde mit Erlös ist eine Komponente und wächst (orange); die Untergrenze steht bei 0.")
    else:
        e = res.events[k - 1]
        if e["kind"] == "merge":
            u, v = e["edge"]
            what = f"die Straße **{u}-{v}** wird fest (die Moats berühren sich), die beiden Komponenten verschmelzen"
        else:
            m = e["members"]
            what = f"eine Komponente mit {len(m)} Knoten **stirbt**: ihr Erlös ist aufgebraucht, sie wächst nicht mehr"
        st.markdown(f"**Ereignis {k} von {smax}** (Zeit {num(e['time'])}): {what}. Wachsende Komponenten: {e['n_active']}; Untergrenze bisher **{num(e['dual'])}**.")
    if smax > 0 and k == smax:
        gw_ = a.sols["gw"]
        st.markdown(f"**Ende des Wachstums:** Untergrenze **{num(a.dual)}**, Zielwert des Optimums {num(a.ref)} ({'exakt' if a.proved else 'bester Fund'}). Der rohe feste Baum durch das Depot hat Zielwert {num(a.sols['gwr'].objective)}; "
                    f"GW-Beschneiden macht daraus {num(gw_.objective)} ({pct(a.gap('gw'))}), starkes Beschneiden {num(a.sols['gws'].objective)} ({pct(a.gap('gws'))}).")
    st.plotly_chart(build_gw_step(inst, g, res, k), width="stretch", key=f"s2_map_{k}")
    st.caption("Orange Kreise = Moats wachsender Komponenten, graue Kreise = Moats von Komponenten, die nicht mehr wachsen (gestorben oder mit dem Depot verbunden); grün = bisher feste Straßen, orange Linie = Straße, die jetzt fest wird, "
               "roter Ring = Komponente, die jetzt stirbt.")
else:
    options = list(C.TREE_OPTIONS)
    if ss.get("tree_widget") not in options:
        ss.pop("tree_widget", None)
    cur = ss["tree_select"] if ss["tree_select"] in options else options[0]
    tree_key = st.radio("Baum zeigen", options=options, format_func=lambda v: C.TREE_LABELS[v], key="tree_widget", horizontal=True, index=options.index(cur), on_change=store_from_widget, args=("tree_select",),
                        help="Bester Fund = niedrigster Zielwert unter allen Verfahren (bei höchstens 12 Kunden das Optimum). Nicht angeschlossene Kunden sind grau und hohl.")
    name = a.best_name if tree_key == "best" else tree_key
    if name not in a.sols:
        st.warning(f"{C.TREE_LABELS.get(tree_key, tree_key)}: das exakte Verfahren wird nur bis t = {C.N_EXACT} Kunden angeboten (hier t = {inst.t}).")
        sol = best
    else:
        sol = a.sols[name]
    st.plotly_chart(build_tree(inst, g, sol), width="stretch", key=f"s3_map_{sol.method}")
    st.markdown(f"**{C.TREE_LABELS.get(sol.method, sol.method)}:** **{len(sol.connected)} von {a.t} Kunden** angeschlossen, Trassenkosten **{num(sol.cost)}**, Erlös der Angeschlossenen {num(sol.net + sol.cost)}, "
                f"**Netto-Erlös {num(sol.net)}**, Zielwert **{num(sol.objective)}**" + ("" if sol.objective <= a.ref + 1e-9 else f" ({pct(a.gap(sol.method))} über {'dem Optimum' if a.proved else 'dem besten Fund'})") +
                f"; {len(sol.steiner)} Steinerknoten, davon {a.n_branch(sol.method)} Verzweigung(en).")
    order = [k for k in ("exact", "ls", "stp", "gws", "gw", "gwr", "all", "none") if k in a.sols]
    bar_order = [k for k in order if k != "gwr"]
    st.plotly_chart(build_net_bars({k: a.sols[k].net for k in bar_order}, bar_order), width="stretch", key="net_bars")
    rows_t = []
    for k in order:
        s = a.sols[k]
        rows_t.append({"Verfahren": METHOD_LABELS[k], "Angeschlossen": f"{len(s.connected)} von {a.t}", "Kosten": round(s.cost, 2), "Netto-Erlös": round(s.net, 2), "Zielwert": round(s.objective, 2),
                       "Lücke Zielwert (%)": round(a.gap(k), 2), "Netto-Verlust (%)": None if a.net_gap(k) is None else round(a.net_gap(k), 1)})
    st.dataframe(pd.DataFrame(rows_t), hide_index=True, width="stretch")
    st.caption(("Lücke gegen das exakte Optimum. " if a.proved else "Lücke gegen den besten gefundenen Baum (bei dieser Größe ohne exaktes Verfahren). ") +
               "Netto-Verlust = wie viel vom Netto-Erlös der Referenz fehlt (leer, wenn der Referenz-Netto-Erlös nicht positiv ist). Kleine Zielwert-Lücken können große Netto-Verluste sein.")

st.markdown("---")

# --- Kennzahlen --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ⚙️ Wer sollte angeschlossen werden?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Netto-Erlös", num(best.net), delta=f"Fund: {C.TREE_LABELS.get(a.best_name, a.best_name)}", delta_color="off")
m2.metric("Angeschlossen", f"{len(best.connected)} von {a.t}", delta=f"Wert der Auswahl {a.select_gain_pct:.1f} %", delta_color="off")
m3.metric("GW (stark)", "optimal" if a.gap("gws") <= 1e-9 else pct(a.gap("gws")), delta="gegen Referenz", delta_color="off")
m4.metric("Erst Steiner", "optimal" if a.gap("stp") <= 1e-9 else pct(a.gap("stp")), delta="gegen Referenz", delta_color="off")
ex_txt = f"Exakt bewiesen ({a.states} Zustände): die Referenz ist das Optimum." if a.proved else f"Kein exaktes Verfahren bei t = {a.t} > {C.N_EXACT}: die Referenz ist der beste Fund."
st.caption(f"Wert der Auswahl = 1 − Referenz-Zielwert / besserer Zielwert von \"alle\" ({num(a.sols['all'].objective)}) und \"nichts\" ({num(a.sols['none'].objective)}). Untergrenze von GW: {num(a.dual)} = {a.dual_ratio * 100:.0f} % der Referenz ({num(a.ref)}). "
           f"Garantie von GW: höchstens {a.bound:.3f} x Optimum (gemessen: GW-Beschneiden {a.sols['gw'].objective / max(a.ref, 1e-12):.3f}, stark {a.sols['gws'].objective / max(a.ref, 1e-12):.3f}). "
           f"Lokalsuche {pct(a.gap('ls'))}, roher GW-Baum {pct(a.gap('gwr'))}. {ex_txt}")

st.markdown("---")

# --- Erlösniveau --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 📈 Vom leeren Baum zu allen Kunden")
with st.spinner("Rechne die Kurve..."):
    rows_c = _curve(replace(settings, level=1.0))
st.plotly_chart(build_level_curve(rows_c, float(level), "Straßenlängen"), width="stretch", key="level_curve")
first = next((r["level"] for r in rows_c if r["opt"]["connected"] > 0), None)
full = next((r["level"] for r in rows_c if r["opt"]["connected"] >= a.t), None)
st.caption(f"Netto-Erlös des Optimums (grün) gegen die Verfahren über das Erlösniveau, Balken = Zahl der angeschlossenen Kunden. Bei diesem Plan wird erst ab Niveau {first:g} überhaupt ein Kunde angeschlossen"
           + (f", ab Niveau {full:g} alle." if full is not None else ", bis Niveau 20 nicht alle.") if first is not None else
           "Bei diesem Plan wird bis Niveau 20 kein Kunde angeschlossen.")

st.markdown("---")

# --- Experimente auf Abruf ---------------------------------------------------------------------------------------------------------------------

base = replace(settings, seed=0)
if kind != "textbook":
    st.subheader("🎲 Wie gut sind die Verfahren?")
    st.caption("50 Instanzen mit den Einstellungen der Seitenleiste (nur der Seed wechselt): wie oft trifft jedes Verfahren die Referenz, wie groß ist die Lücke, wie weit liegt die Untergrenze unter dem Optimum?")
    if st.button("Qualitäts-Experiment über 50 Instanzen (dauert einige Sekunden)", key="quality_start"):
        ss["quality_done"] = ss.get("quality_done", set()) | {base}
    if base in ss.get("quality_done", set()):
        with st.spinner("Rechne..."):
            q = _quality(base)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Auswahl lohnt", f"{q['select_share']:.0f} %", delta=f"im Mittel {q['select_gain_mean']:.1f} %", delta_color="off")
        q2.metric("GW (stark) optimal", f"{q['gws_optimal']:.0f} %", delta=f"mittlere Lücke {q['gws_gap_mean']:.2f} %", delta_color="off")
        q3.metric("Erst Steiner optimal", f"{q['stp_optimal']:.0f} %", delta=f"mittlere Lücke {q['stp_gap_mean']:.2f} %", delta_color="off")
        q4.metric("Lokalsuche optimal", f"{q['ls_optimal']:.0f} %", delta=f"mittlere Lücke {q['ls_gap_mean']:.2f} %", delta_color="off")
        st.caption(f"Über {q['n_runs']} Instanzen; " + ("Lücken gegen das exakte Optimum. " if q["exact_used"] else "Lücken gegen den besten gefundenen Baum (bei dieser Größe ohne exaktes Verfahren). ")
                   + f"GW mit GW-Beschneiden ist in {q['gw_optimal']:.0f} % optimal (mittlere Lücke {q['gw_gap_mean']:.2f} %, größte {q['gw_gap_max']:.1f} %), stark beschnitten größte Lücke {q['gws_gap_max']:.1f} %; "
                   f"starkes Beschneiden ist besser als GW-Beschneiden in {q['gws_better_than_gw']:.0f} %. \"Erst Steiner, dann kürzen\" schlägt GW (stark) in {q['stp_beats_gws']:.0f} %, GW (stark) schlägt es in {q['gws_beats_stp']:.0f} %. "
                   f"Untergrenze im Mittel {q['dual_ratio_mean'] * 100:.0f} % des Optimums (kleinstes {q['dual_ratio_min'] * 100:.0f} %). Mittlerer Netto-Verlust: GW (stark) {q['gws_net_gap_mean']:.1f} % bei Zielwert-Lücke {q['gws_gap_mean']:.2f} %. "
                   f"Alle anschließen: {q['all_gap_mean']:.1f} % Lücke, nichts anschließen: {q['none_gap_mean']:.1f} %. Garantieverletzungen: {q['guarantee_violations']}.")
    st.markdown("---")

    st.subheader("🪆 Wächst die Menge der angeschlossenen Kunden nur?")
    st.caption("Je Instanz alle Erlösniveaus der Regler nacheinander: ist der optimale Baum der höheren Stufe immer ein Obermengen-Baum der niedrigeren (die Anschlussmenge \"genestet\")? "
               "Der gesammelte Erlös des Optimums fällt nie; die Menge selbst darf wechseln, weil die Kosten des Steinerbaums keine einfache Summe der Kunden sind.")
    if st.button("Nestungs-Experiment über 50 Instanzen (dauert einige Sekunden)", key="nesting_start"):
        ss["nesting_done"] = ss.get("nesting_done", set()) | {base}
    if base in ss.get("nesting_done", set()):
        with st.spinner("Rechne..."):
            nn = _nesting(base)
        n1, n2 = st.columns(2)
        n1.metric("Nestung verletzt", f"{nn['violated']} von {nn['n_runs']}", delta=f"{nn['violated_share']:.0f} %" if nn["n_runs"] else "", delta_color="off")
        n2.metric("Erlös fällt", str(nn["prize_drops"]), delta="Stufenwechsel", delta_color="off")
        st.caption("Beispiele (Seed, Wechsel der Stufen): " + (", ".join(f"Seed {x['seed']}: {x['from']:g} → {x['to']:g}" for x in nn["examples"][:5]) if nn["examples"] else "keine Verletzung bei diesen Einstellungen.") +
                   f" Nur Instanzen mit höchstens {C.N_EXACT} Kunden (exakte Rechnung).")
    st.markdown("---")

    st.subheader("⏱️ Was kostet der exakte Weg?")
    st.caption("Dreyfus-Wagner über die Kundenzahl auf demselben Plan: die Zahl der Zustände (Teilmenge, Kreuzung) wächst mit 2^t, der Rechenaufwand mit 3^t.")
    if st.button("Aufwandskurve berechnen (dauert einige Sekunden)", key="time_start"):
        ss["time_done"] = ss.get("time_done", set()) | {base}
    if base in ss.get("time_done", set()):
        with st.spinner("Rechne..."):
            rows_tt = _time_curve(base)
        st.plotly_chart(build_time_curve(rows_tt), width="stretch", key="time_curve")
        st.caption("Die Sekunden sind eine Messung auf diesem Rechner und schwanken; die Zustandszahlen sind exakt.")
    st.markdown("---")

    st.subheader("📐 Sweeps")
    sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda v: SWEEP_LABELS[v], key="sweep_select")
    metric_opts = {"gap": "Zielwert-Lücke der Verfahren", "net": "Netto-Verlust der Verfahren", "select": "Wert der Auswahl und angeschlossene Kunden", "dual": "Untergrenze / Referenz"}
    if ss.get("sweep_metric") not in metric_opts:
        ss.pop("sweep_metric", None)
    metric = st.radio("Kennzahl", options=list(metric_opts), format_func=lambda v: metric_opts[v], key="sweep_metric", horizontal=True)
    if st.button("Sweep über 5 feste Instanzen berechnen (kann einige Sekunden dauern)", key="sweep_start"):
        ss["sweep_done"] = ss.get("sweep_done", set()) | {(sweep_param, base)}
    if (sweep_param, base) in ss.get("sweep_done", set()):
        with st.spinner("Rechne den Sweep über 5 feste Instanzen..."):
            rows_s = _sweep(sweep_param, base)
        series = {
            "gap": ([("gap_gw", "GW (GW-Beschneiden)", "#b39ddb"), ("gap_gws", "GW (stark)", "#7b3fbf"), ("gap_stp", "Erst Steiner, dann kürzen", "#4c78a8"), ("gap_ls", "Lokalsuche", "#e8a13a")], "Lücke gegen die Referenz (%)"),
            "net": ([("net_gap_gw", "GW (GW-Beschneiden)", "#b39ddb"), ("net_gap_gws", "GW (stark)", "#7b3fbf"), ("net_gap_stp", "Erst Steiner, dann kürzen", "#4c78a8"), ("net_gap_ls", "Lokalsuche", "#e8a13a")], "Netto-Verlust gegen die Referenz (%)"),
            "select": ([("select_gain_pct", "Wert der Auswahl (%)", "#2F6B65"), ("connected_share", "angeschlossene Kunden (%)", "#4c78a8")], "Prozent"),
            "dual": ([("dual_ratio", "Untergrenze / Referenz", "#7b3fbf")], "Verhältnis"),
        }[metric]
        st.plotly_chart(build_sweep(rows_s, SWEEP_LABELS[sweep_param], series[0], series[1], tick=SWEEP_TICKS.get(sweep_param)), width="stretch", key="sweep_chart")
        st.caption("Median über 5 feste Instanzen (Seeds 100000–100004), Band = 10. bis 90. Perzentil. Die Lücke ist gegen die Referenz gemessen - bei bis zu 12 Kunden das exakte Optimum. Die übrigen Regler stehen wie in der Seitenleiste.")
    st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Auswahl lohnt sich immer** | Nur mitten im Übergang: bei Niveau 2.5 (Plan 8 x 8, 8 Kunden) bringt das Optimum gegenüber der besseren Grundlinie "alle" oder "nichts" in 90 % der Instanzen etwas, im Mittel 6.0 %; bei Niveau 1.5 in 18 % (1.1 %), bei Niveau 4 in 58 % (1.8 %). In Gruppen liegt der Übergang tiefer (Niveau 1.5: 78 %, Niveau 2.5: 10 %); bei 30 % gesperrten Straßen werden im Mittel 3.0 statt 6.3 von 8 Kunden angeschlossen. | - |
| **GW ist nah am Optimum** | Nein: bei Niveau 2.5 ist GW mit GW-Beschneiden nie optimal (mittlere Lücke 12.8 %, größte 28.4 %), mit starkem Beschneiden in 30 % (2.83 %, größte 12.4 %); starkes Beschneiden ist in 88 % der Instanzen besser. Bei Niveau 1.5 kostet das grobe Beschneiden 22.0 % (stark: 0), bei Niveau 4 liegen beide bei 6 bis 7 % - GW ist als Steinerbaum-Verfahren schwach. Die Garantie 2 − 1/(n − 1) wurde in keiner gemessenen Instanz verletzt. | Erst Steiner, dann kürzen; Lokalsuche |
| **GW ist die richtige Wahl** | "Erst Steinerbaum, dann kürzen" liegt bei Niveau 2.5 im Mittel nur 0.83 % über dem Optimum (in 56 % optimal), die Lokalsuche 0.69 % (60 %); es schlägt GW (stark) in 58 % der Instanzen, GW schlägt es in 8 %. GW liefert dafür eine **Untergrenze** (im Mittel 68 % des Optimums) und eine bewiesene Garantie, die die anderen nicht haben. | - |
| **Kleine Zielwert-Lücke, kleiner Verlust** | Nein: bei Niveau 2.5 sind 2.83 % Zielwert-Lücke von GW (stark) im Mittel 24.6 % Netto-Erlös-Verlust; bei "erst Steiner" 0.83 % → 9.7 %, bei GW mit GW-Beschneiden 12.8 % → 163.8 %. Die Garantie gilt für den Zielwert, nicht für den Netto-Erlös. | Verfahren mit Netto-Erlös-Garantie (nicht gebaut) |
| **Die Anschlussmenge wächst mit dem Niveau** | Der gesammelte Erlös des Optimums fällt nie (in allen Messungen 0 Ausnahmen), die Menge selbst wechselt aber manchmal: auf dem Plan 8 x 8 in 0 von 50 Instanzen, auf 10 x 10 mit 10 Kunden in 2 von 50; im Preset (Seed 12) fällt Kunde 58 heraus, obwohl sein Erlös steigt. Ein Gegenbeispiel auf 6 Knoten ist per Brute-Force bestätigt. | - |
| **Exakt lösbar** | Nur bis 12 Kunden (Dreyfus-Wagner, Aufwand etwa 3^t; t = 12 dauert rund 2 s); darüber vergleicht die Demo die Verfahren untereinander (Referenz = bester Fund). Stand der Technik: 1.79-Approximation (Ahmadi u. a. 2024), Branch-and-Cut - beides nicht gebaut. | Branch-and-Cut, Reduktionen |
| **Synthetisches Modell** | Gestörtes Gitter, Länge als Kosten, Erlöse additiv und unabhängig, ein Depot (gewurzelt); die ungewurzelte Variante, Mindestmengen und Kapazitäten kommen nicht vor. | Echte Netze |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gegeben ein zusammenhängender Graph $G = (V, E)$ mit Längen $c_e > 0$, ein Depot $r \in V$ und Erlöse $\pi_v \ge 0$. Gesucht ist ein Baum $F \subseteq E$ durch $r$ mit minimalem Zielwert
$c(F) + \sum_{v \notin V(F)} \pi_v$. Der Netto-Erlös $\sum_{v \in V(F)} \pi_v - c(F)$ ist Gesamterlös minus Zielwert.

**Kernsatz.** $\mathrm{OPT} = \min_{r \in X \subseteq V} \mathrm{MST}\big(G[X]\big) + \pi(V \setminus X)$.

**Exakt.** Dreyfus-Wagner über die Kunden $K$: $D[S][v]$ = kürzester Baum, der die Kunden in $S$ und den Knoten $v$ verbindet; $\mathrm{OPT} = \min_{S \subseteq K} D[S][r] + \pi(K \setminus S)$. Aufwand $O(3^{t} n + 2^{t} n^2)$.

**Goemans-Williamson.** Dualvariablen $y_S \ge 0$ für Knotenmengen $S \subseteq V \setminus \{r\}$ mit $\sum_{S \subseteq C} y_S \le \pi(C)$ für jede Menge $C$ und $\sum_{S:\, e \in \delta(S)} y_S \le c_e$ für jede Kante $e$ ($\delta(S)$ = Kanten, die $S$ verlassen). Der Wert
$\sum_S y_S$ ist eine Untergrenze des Optimums; das Verfahren erreicht höchstens $2 - 1/(n-1)$ mal das Optimum.

**Literatur.** Goemans, M. X., & Williamson, D. P. (1995). *A general approximation technique for constrained forest problems.* SIAM Journal on Computing 24(2), 296-317. Johnson, D. S., Minkoff, M., & Phillips, S. (2000).
*The prize collecting Steiner tree problem: theory and practice.* Proc. 11th ACM-SIAM Symposium on Discrete Algorithms, 760-769. Ahmadi, A., Gholami, I., Hajiaghayi, M., Jabbarzade, P., & Mahdavi, M. (2024). *Prize-collecting Steiner tree: a 1.79
approximation.* STOC 2024 (nur genannt, nicht gebaut). Bausteine des Steinerbaums: Kou, Markowsky & Berman (1981), Takahashi & Matsuyama (1980), Dreyfus & Wagner (1971) wie in der Steiner-Baum-Demo.

Implementiert in `pcst_algorithm.py` (Verfahren), `pcst_scenario.py` (Pläne), `pcst_evaluation.py` (Kennzahlen, Sweeps, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
