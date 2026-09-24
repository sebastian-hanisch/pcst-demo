"""Plotly-Abbildungen: Stadtplan mit Kunden (Größe = Erlös) und gesperrten Straßen, Baum mit angeschlossenen und nicht angeschlossenen Kunden (Steinerpunkte als Rauten), Goemans-Williamson Ereignis für
Ereignis (Moats als Kreise), Erlösniveau-Kurve, Netto-Erlös-Balken, Aufwandskurve, Sweeps. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

from collections import Counter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pcst_constants as C
from pcst_algorithm import branch_points

TREE_COLOR = "#2F6B65"
STEINER_COLOR = "#ff7f0e"
CUSTOMER_COLOR = "#4c78a8"
ROOT_COLOR = "#2ca02c"
OUT_COLOR = "#9a9a9a"
BLOCKED_COLOR = "rgba(214,39,40,0.35)"
STREET_COLOR = "rgba(150,150,150,0.35)"
ACTIVE_COLOR = "rgba(255,127,14,0.22)"
DEAD_COLOR = "rgba(120,120,120,0.16)"
METHOD_COLORS = {"exact": "#2F6B65", "ls": "#e8a13a", "stp": "#4c78a8", "gws": "#7b3fbf", "gw": "#b39ddb", "gwr": "#9e9e9e", "all": "#8c8c8c", "none": "#c0c0c0"}
METHOD_LABELS = {"exact": "Exakt", "ls": "Lokalsuche", "stp": "Erst Steiner, dann kürzen", "gws": "GW (stark beschnitten)", "gw": "GW (GW-beschnitten)", "gwr": "GW (roh)", "all": "Alle anschließen", "none": "Nichts anschließen"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.1):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_axes(fig, inst, height=460, pad=None):
    """Karte mit gleichem Maßstab in x und y. Ohne feste Bereiche (Plotly rechnet sie beim ersten Zeichnen in kleiner Breite um und behält sie dann); `pad` weitet den Rahmen über zwei unsichtbare Eckpunkte."""
    if pad:
        xs = [p[0] for p in inst.xy]
        ys = [p[1] for p in inst.xy]
        fig.add_trace(go.Scatter(x=[min(xs) - pad, max(xs) + pad], y=[min(ys) - pad, max(ys) + pad], mode="markers", marker=dict(opacity=0, size=1), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, autorange="reversed")
    return _base(fig, height)


def _height(inst):
    return 340 if inst.kind == "textbook" else 480


def _unit(inst):
    return C.SPACING if inst.kind == "city" else 1.0


def _lines(fig, inst, pairs, color, width=2.6, dash="solid", name="", showlegend=False):
    pairs = list(pairs)
    if not pairs:
        return
    xs, ys = [], []
    for u, v in pairs:
        xs += [inst.xy[u][0], inst.xy[v][0], None]
        ys += [inst.xy[u][1], inst.xy[v][1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), name=name, hoverinfo="skip", showlegend=showlegend))


def _streets(fig, inst, blocked=True):
    _lines(fig, inst, [(u, v) for u, v, _w in inst.edges], STREET_COLOR, 1.0)
    if blocked:
        _lines(fig, inst, inst.blocked_edges, BLOCKED_COLOR, 1.2, "dot", "gesperrt", bool(inst.blocked_edges))


def _size(inst, v):
    top = max(inst.prizes) if max(inst.prizes) > 0 else 1.0
    return 10 + 16 * (inst.prizes[v] / top)


def _prize_text(inst, v):
    return f"{inst.prizes[v] / _unit(inst):.1f}"


def _customers(fig, inst, connected=None, labels=True, on_label="Kunde (angeschlossen)"):
    """Kunden als Kreise, Größe = Erlös; angeschlossene blau, nicht angeschlossene grau und hohl. Beschriftung = Erlös in Straßenlängen (bei bis zu 12 Kunden)."""
    cs = list(inst.customers)
    connected = set(cs) if connected is None else set(connected)
    on = [v for v in cs if v in connected]
    off = [v for v in cs if v not in connected]
    text = labels and len(cs) <= 12
    if on:
        fig.add_trace(go.Scatter(x=[inst.xy[v][0] for v in on], y=[inst.xy[v][1] for v in on], mode="markers+text" if text else "markers", text=[_prize_text(inst, v) for v in on] if text else None,
                                 textposition="top center", marker=dict(size=[_size(inst, v) for v in on], color=CUSTOMER_COLOR, line=dict(width=1.5, color="white")), name=on_label, hoverinfo="skip", showlegend=True))
    if off:
        fig.add_trace(go.Scatter(x=[inst.xy[v][0] for v in off], y=[inst.xy[v][1] for v in off], mode="markers+text" if text else "markers", text=[_prize_text(inst, v) for v in off] if text else None,
                                 textposition="top center", marker=dict(size=[_size(inst, v) for v in off], color="rgba(255,255,255,0.9)", line=dict(width=2.2, color=OUT_COLOR)), name="Kunde (nicht angeschlossen)",
                                 hoverinfo="skip", showlegend=True))
    fig.add_trace(go.Scatter(x=[inst.xy[inst.depot][0]], y=[inst.xy[inst.depot][1]], mode="markers", marker=dict(size=18, symbol="star", color=ROOT_COLOR, line=dict(width=1, color="white")),
                             name="Depot", hoverinfo="skip", showlegend=True))


def _steiner(fig, inst, nodes, branch=None, names=("Verzweigung (Steinerpunkt)", "Durchgang")):
    """Steinerpunkte: Rauten; Verzweigungspunkte (Grad >= 3) orange gefüllt, Durchgangsknoten (Grad 2) hohl."""
    nodes = list(nodes)
    if not nodes:
        return
    branch = set(branch) if branch is not None else set(nodes)
    b = [v for v in nodes if v in branch]
    p = [v for v in nodes if v not in branch]
    if b:
        fig.add_trace(go.Scatter(x=[inst.xy[v][0] for v in b], y=[inst.xy[v][1] for v in b], mode="markers", marker=dict(size=13, symbol="diamond", color=STEINER_COLOR, line=dict(width=1.5, color="white")),
                                 name=names[0], hoverinfo="skip", showlegend=True))
    if p:
        fig.add_trace(go.Scatter(x=[inst.xy[v][0] for v in p], y=[inst.xy[v][1] for v in p], mode="markers", marker=dict(size=8, symbol="diamond-open", color=STEINER_COLOR, line=dict(width=1.5)),
                                 name=names[1], hoverinfo="skip", showlegend=True))


def build_instance(inst):
    fig = go.Figure()
    _streets(fig, inst)
    _customers(fig, inst, on_label="Kunde")
    return _map_axes(fig, inst, _height(inst))


def build_tree(inst, g, sol):
    """Der Baum in Grün; angeschlossene Kunden blau, nicht angeschlossene grau und hohl, Steinerpunkte als Rauten."""
    fig = go.Figure()
    _streets(fig, inst, blocked=False)
    _lines(fig, inst, sol.edges, TREE_COLOR, 3.6, name="Baum", showlegend=bool(sol.edges))
    _steiner(fig, inst, sol.steiner, branch_points(g, sol.edges))
    _customers(fig, inst, sol.connected)
    return _map_axes(fig, inst, _height(inst))


def build_gw_step(inst, g, res, k):
    """Goemans-Williamson nach `k` Ereignissen: die Moats (Kreise um jeden Knoten mit Radius d_v; orange = wachsende Komponente, grau = Komponente ohne Wachstum), die bisher festen Kanten (grün) und das
    Ereignis selbst (Kante orange bzw. sterbende Komponente rot umrandet)."""
    ev = res.events
    n = inst.n
    if k <= 0:
        comp, active, d, merges = list(range(n)), [1 if (v != inst.depot and inst.prizes[v] > 1e-9) else 0 for v in range(n)], [0.0] * n, 0
    else:
        e = ev[k - 1]
        comp, active, d = e["comp"], e["active"], e["d"]
        merges = sum(1 for x in ev[:k] if x["kind"] == "merge")
    fig = go.Figure()
    _streets(fig, inst, blocked=False)
    shapes = []
    for v in range(n):
        if d[v] > 1e-9:
            r = d[v]
            shapes.append(dict(type="circle", xref="x", yref="y", x0=inst.xy[v][0] - r, x1=inst.xy[v][0] + r, y0=inst.xy[v][1] - r, y1=inst.xy[v][1] + r, fillcolor=ACTIVE_COLOR if active[v] else DEAD_COLOR,
                               line=dict(width=0), layer="below"))
    fig.update_layout(shapes=shapes)
    forest = res.forest[:merges]
    _lines(fig, inst, forest, TREE_COLOR, 3.6, name="feste Kanten", showlegend=bool(forest))
    if k >= 1:
        e = ev[k - 1]
        if e["kind"] == "merge":
            _lines(fig, inst, [tuple(sorted(e["edge"]))], STEINER_COLOR, 5.0, name="Kante wird fest", showlegend=True)
        else:
            m = e["members"]
            fig.add_trace(go.Scatter(x=[inst.xy[v][0] for v in m], y=[inst.xy[v][1] for v in m], mode="markers", marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=2.6, color="#d62728")),
                                     name="Komponente stirbt", hoverinfo="skip", showlegend=True))
    size = Counter(comp)
    inner = [v for v in range(n) if v != inst.depot and v not in inst.customers and size[comp[v]] > 1]
    _steiner(fig, inst, inner, set(), names=("", "Steinerknoten in einer Komponente"))
    _customers(fig, inst, None, labels=True, on_label="Kunde")
    max_d = max((max(x["d"]) for x in ev), default=0.0)
    return _map_axes(fig, inst, _height(inst), pad=min(max_d, C.SPACING * 2.0 if inst.kind == "city" else 1.6))


def build_level_curve(rows, current, unit_label="Straßenlängen"):
    """Netto-Erlös in Prozent des Gesamterlöses je Verfahren über das Erlösniveau (Linien) und Zahl der im Optimum angeschlossenen Kunden (Balken, rechte Achse). "Nichts anschließen" ist die Nulllinie."""
    xs = [f"{r['level']:g}" for r in rows]

    def share(r, k):
        return 100.0 * r[k]["net"] / r["total_prize"] if r["total_prize"] > 0 else 0.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=xs, y=[r["opt"]["connected"] for r in rows], marker_color="rgba(76,120,168,0.18)", name="angeschlossen (Optimum / bester Fund)"), secondary_y=True)
    for key, dash, width in (("all", "dash", 2.0), ("gw", "solid", 1.8), ("gws", "solid", 2.2), ("stp", "solid", 2.2)):
        if key in rows[0]:
            fig.add_trace(go.Scatter(x=xs, y=[share(r, key) for r in rows], mode="lines", line=dict(color=METHOD_COLORS[key], width=width, dash=dash), name=METHOD_LABELS[key]), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=[share(r, "opt") for r in rows], mode="lines+markers", line=dict(color=METHOD_COLORS["exact"], width=3.4), name="Optimum / bester Fund"), secondary_y=False)
    fig.add_hline(y=0, line=dict(color=METHOD_COLORS["none"], width=1.5, dash="dot"), secondary_y=False)
    cur = [r for r in rows if abs(r["level"] - current) < 1e-9]
    if cur:
        fig.add_trace(go.Scatter(x=[f"{cur[0]['level']:g}"], y=[share(cur[0], "opt")], mode="markers", marker=dict(size=16, symbol="diamond", color="#d62728", line=dict(width=1.5, color="white")), name="aktuelles Niveau"),
                      secondary_y=False)
    fig.update_xaxes(title_text=f"Erlösniveau ({unit_label} je Kunde)", type="category")
    fig.update_yaxes(title_text="Netto-Erlös in % des Gesamterlöses", secondary_y=False, range=[-50, 105])
    fig.update_yaxes(title_text="angeschlossene Kunden", secondary_y=True, showgrid=False, range=[0, rows[0]["t"]], dtick=1)
    return _base(fig, 380, legend_y=-0.35)


def build_net_bars(nets, order):
    """Netto-Erlös je Verfahren (Balken; "nichts anschließen" = 0)."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[METHOD_LABELS[k] for k in order], y=[nets[k] for k in order], marker_color=[METHOD_COLORS[k] for k in order], text=[f"{nets[k]:.1f}" for k in order], textposition="outside"))
    fig.update_yaxes(title_text="Netto-Erlös")
    return _base(fig, 330)


def build_time_curve(rows):
    """Aufwand des exakten Wegs über die Kundenzahl: Zustände (Balken) und gemessene Sekunden (Linie), beides logarithmisch."""
    xs = [str(r["t"]) for r in rows]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=xs, y=[r["states"] for r in rows], marker_color="rgba(76,120,168,0.25)", name="Zustände"), secondary_y=False)
    fig.add_trace(go.Scatter(x=xs, y=[max(r["seconds"], 1e-4) for r in rows], mode="lines+markers", line=dict(color="#d62728", width=2.6), name="Sekunden (gemessen)"), secondary_y=True)
    fig.update_xaxes(title_text="Kunden t", type="category")
    fig.update_yaxes(title_text="Zustände (Teilmenge, Knoten)", type="log", secondary_y=False)
    fig.update_yaxes(title_text="Sekunden", type="log", secondary_y=True, showgrid=False)
    return _base(fig, 340, legend_y=-0.3)


def build_sweep(rows, param_label, series, y_label, tick=None, log_y=False):
    """`series` = [(key, Name, Farbe)]: Median als Linie, 10. bis 90. Perzentil als Band (`<key>_lo`/`<key>_hi`)."""
    xs = [tick(r["value"]) if tick else str(r["value"]) for r in rows]
    fig = go.Figure()
    for key, name, color in series:
        ys = [None if r[key] != r[key] else r[key] for r in rows]
        lo = [None if r.get(f"{key}_lo", r[key]) != r.get(f"{key}_lo", r[key]) else r.get(f"{key}_lo", r[key]) for r in rows]
        hi = [None if r.get(f"{key}_hi", r[key]) != r.get(f"{key}_hi", r[key]) else r.get(f"{key}_hi", r[key]) for r in rows]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        if all(v is not None for v in lo + hi):
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.13)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5), name=name, connectgaps=False))
    fig.update_xaxes(title_text=param_label, type="category")
    fig.update_yaxes(title_text=y_label, type="log" if log_y else "linear")
    return _base(fig, 360, legend_y=-0.3)
