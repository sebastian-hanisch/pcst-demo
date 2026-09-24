"""Auswertung: was bringt die Auswahl der Kunden, und wie gut sind die Verfahren?

Verfahren auf derselben Instanz: Goemans-Williamson (GW roh, mit GW-Beschneiden, mit starkem Beschneiden), "erst Steinerbaum über alle Kunden, dann optimal kürzen" (STP), Lokalsuche über Knotenmengen (vom besseren Start
aus GW streng / STP) und - für wenige Kunden - exakt (Dreyfus-Wagner über die Kunden). Dazu die Grundlinien "alle anschließen" (bester gefundener Steinerbaum über alle) und "nichts anschließen" (nur das Depot).
Alles ist deterministisch: Kennzahlen laufen über 5 feste Instanzen (Seeds 100000-100004), Median mit 10./90. Perzentil.

- **Zielwert** = Trassenkosten + Erlöse der nicht angeschlossenen Kunden (kleiner ist besser); **Netto-Erlös** = Erlöse der angeschlossenen Kunden - Trassenkosten (`net`).
- **Referenz** (`ref`) = Zielwert des exakten Optimums (t <= N_EXACT), sonst des besten gefundenen Baums.
- **Lücke** (`gap_*`) = Zielwert des Verfahrens / Referenz - 1 in Prozent; **Netto-Lücke** (`net_gap_*`) = Netto-Erlös-Verlust gegen die Referenz in Prozent des Referenz-Netto-Erlöses (nur wo dieser positiv ist).
- **Wert der Auswahl** (`select_gain_pct`) = 1 - Referenz / besserer Zielwert von "alle" und "nichts", in Prozent.
- **Dual-Verhältnis** (`dual_ratio`) = GW-Untergrenze / Referenz (<= 1, die Lücke zwischen Untergrenze und Optimum)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import pcst_algorithm as A
import pcst_constants as C
import pcst_scenario as S

INF = float("inf")
SOLUTION_NAMES = ("exact", "ls", "stp", "gws", "gw", "gwr", "all", "none")


@dataclass(frozen=True)
class Settings:
    kind: str = "city"
    side: int = C.DEFAULT_SIDE
    t: int = C.DEFAULT_T
    blocked: float = C.DEFAULT_BLOCKED
    layout: str = "uniform"
    prize_mode: str = "equal"
    level: float = C.DEFAULT_LEVEL
    seed: int = C.DEFAULT_SEED


@lru_cache(maxsize=256)
def instance_of(settings):
    if settings.kind == "textbook":
        return S.textbook_instance(settings.level)
    return S.generate(settings.side, min(settings.t, settings.side * settings.side - 1), settings.blocked, settings.layout, settings.prize_mode, settings.level, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    g: object                          # Graph
    res: object                        # GWResult
    sols: dict                         # Name -> Solution (exact nur bei t <= N_EXACT)
    exact_offered: bool = False
    states: int = 0
    tables: object = None              # ExactTables

    @property
    def t(self):
        return self.inst.t

    @property
    def dual(self):
        return self.res.dual

    @property
    def best_name(self):
        return min(self.sols, key=lambda k: (round(self.sols[k].objective, 9), SOLUTION_NAMES.index(k)))

    @property
    def best(self):
        return self.sols[self.best_name]

    @property
    def proved(self):
        return "exact" in self.sols

    @property
    def ref(self):
        return self.sols["exact"].objective if self.proved else self.best.objective

    @property
    def ref_net(self):
        return self.inst.total_prize - self.ref

    @property
    def bound(self):
        """Garantiefaktor 2 - 1/(n-1) von Goemans-Williamson (n Knoten)."""
        return 2.0 - 1.0 / (self.inst.n - 1) if self.inst.n > 1 else 1.0

    @property
    def dual_ratio(self):
        return self.dual / self.ref if self.ref > 1e-12 else 1.0

    def gap(self, name):
        s = self.sols.get(name)
        if s is None:
            return None
        return 0.0 if self.ref <= 1e-12 or s.objective <= self.ref + 1e-9 else 100.0 * (s.objective / self.ref - 1.0)

    def net_gap(self, name):
        s = self.sols.get(name)
        if s is None or self.ref_net <= 1e-9:
            return None
        return max(0.0, 100.0 * (self.ref_net - s.net) / self.ref_net)

    @property
    def select_gain_pct(self):
        base = min(self.sols["all"].objective, self.sols["none"].objective)
        return 0.0 if base <= 1e-12 or self.ref >= base - 1e-9 else 100.0 * (1.0 - self.ref / base)

    def n_branch(self, name):
        return len(A.branch_points(self.g, self.sols[name].edges))


def exact_offered(inst):
    return inst.t <= C.N_EXACT


def analyse(settings):
    inst = instance_of(settings)
    g = A.Graph(inst)
    res, gw = A.gw_solutions(g)
    sols = dict(gw)
    sols["all"] = A.steiner_all(g)
    sols["stp"] = A.steiner_then_prune(g, sols["all"])
    sols["none"] = A.connect_none(g)
    start = min((sols["gws"], sols["stp"]), key=lambda x: x.objective)
    sols["ls"] = A.local_search(g, start.edges)
    a = Analysis(settings, inst, g, res, sols, exact_offered=exact_offered(inst))
    if a.exact_offered:
        ex = A.pcst_exact(g)
        sols["exact"] = ex.solution
        a.states = ex.states
        a.tables = ex.tables
    return a


# --- Kennzahlen über feste Instanzen ------------------------------------------------------------------------------------------------------------


def _stats(values):
    values = [v for v in values if v is not None and not np.isnan(v) and v != INF]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90))


METHODS = ("gw", "gws", "gwr", "stp", "ls", "all", "none")


def run_config(base, seeds=C.SWEEP_SEEDS, **changes):
    s0 = replace(base, **changes)
    rows = [analyse(replace(s0, seed=seed)) for seed in seeds]
    out = {"n_runs": len(rows), "offered_share": 100.0 * sum(r.exact_offered for r in rows) / len(rows),
           "guarantee_violations": sum(1 for r in rows for k in ("gw", "gws") if r.ref > 1e-12 and r.sols[k].objective > r.bound * r.ref + 1e-9)}
    for nm in ("gw", "gws", "stp", "ls"):
        out[f"{nm}_optimal_share"] = 100.0 * sum(1 for r in rows if r.gap(nm) <= 1e-9) / len(rows)
    cols = [("ref", [r.ref for r in rows]), ("ref_net", [r.ref_net for r in rows]), ("dual_ratio", [r.dual_ratio for r in rows]), ("select_gain_pct", [r.select_gain_pct for r in rows]),
            ("connected", [float(len(r.best.connected)) for r in rows]), ("connected_share", [100.0 * len(r.best.connected) / r.t for r in rows]),
            ("cost", [r.best.cost for r in rows]), ("n_branch", [float(r.n_branch(r.best_name)) for r in rows]), ("events", [float(len(r.res.events)) for r in rows])]
    for nm in METHODS:
        cols.append((f"gap_{nm}", [r.gap(nm) for r in rows]))
        cols.append((f"net_gap_{nm}", [r.net_gap(nm) for r in rows]))
    for key, values in cols:
        out[key], out[f"{key}_lo"], out[f"{key}_hi"] = _stats(values)
    return out


SWEEP_VALUES = {"level": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0), "t": (3, 5, 8, 10, 12, 20, 30), "side": (5, 6, 8, 10, 12, 14), "blocked": C.BLOCKED_OPTIONS,
                "layout": C.LAYOUTS, "prize_mode": C.PRIZE_MODES}
SWEEP_LABELS = {"level": "Erlösniveau (Straßenlängen)", "t": "Kunden t", "side": "Gittergröße", "blocked": "Gesperrter Anteil", "layout": "Lage der Kunden", "prize_mode": "Erlösmodus"}
SWEEP_TICKS = {"blocked": lambda v: f"{v:.0%}", "layout": lambda v: C.LAYOUT_LABELS[v], "prize_mode": lambda v: C.PRIZE_LABELS[v]}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def quality(base, seeds=C.FEAS_SEEDS):
    """Wie gut sind die Verfahren? Über `seeds` Instanzen mit den Einstellungen von `base` (nur der Seed wechselt): Anteil, in dem jedes Verfahren die Referenz trifft, mittlere und größte Lücke,
    mittleres Dual-Verhältnis, mittlere Netto-Lücke gegen mittlere Zielwert-Lücke von GW, Garantieverletzungen (müssen 0 sein), Anteil der Instanzen, in denen die Auswahl etwas bringt."""
    an = [analyse(replace(base, seed=seed)) for seed in seeds]
    m = len(an)
    out = {"n_runs": m, "exact_used": all(a.proved for a in an), "dual_ratio_mean": float(np.mean([a.dual_ratio for a in an])), "dual_ratio_min": float(min(a.dual_ratio for a in an)),
           "select_share": 100.0 * sum(a.select_gain_pct > 1e-9 for a in an) / m, "select_gain_mean": float(np.mean([a.select_gain_pct for a in an])),
           "guarantee_violations": sum(1 for a in an for k in ("gw", "gws") if a.ref > 1e-12 and a.sols[k].objective > a.bound * a.ref + 1e-9),
           "connected_mean": float(np.mean([len(a.best.connected) for a in an]))}
    for nm in ("gw", "gws", "gwr", "stp", "ls", "all", "none"):
        gaps = [a.gap(nm) for a in an]
        out[f"{nm}_optimal"] = 100.0 * sum(g <= 1e-9 for g in gaps) / m
        out[f"{nm}_gap_mean"] = float(np.mean(gaps))
        out[f"{nm}_gap_max"] = float(max(gaps))
        ng = [a.net_gap(nm) for a in an]
        ng = [x for x in ng if x is not None]
        out[f"{nm}_net_gap_mean"] = float(np.mean(ng)) if ng else float("nan")
    out["stp_beats_gws"] = 100.0 * sum(a.sols["stp"].objective < a.sols["gws"].objective - 1e-9 for a in an) / m
    out["gws_beats_stp"] = 100.0 * sum(a.sols["gws"].objective < a.sols["stp"].objective - 1e-9 for a in an) / m
    out["gws_better_than_gw"] = 100.0 * sum(a.sols["gws"].objective < a.sols["gw"].objective - 1e-9 for a in an) / m
    out["ls_improves"] = 100.0 * sum(a.sols["ls"].objective < min(a.sols["gws"].objective, a.sols["stp"].objective) - 1e-9 for a in an) / m
    return out


# --- Erlösniveau: leerer Baum <-> alle Kunden -----------------------------------------------------------------------------------------------------


def level_curve(base, levels=C.LEVELS):
    """Über das Erlösniveau (gleicher Plan, gleiche Kunden, Erlöse skalieren linear; im Lehrbuchbeispiel zahlen A und B das Niveau): je Niveau die Zielwerte und angeschlossenen Kunden von exakt (falls
    t <= N_EXACT, sonst bester Fund), GW (GW-Beschneiden und stark), "erst Steinerbaum, dann kürzen", "alle" und "nichts"."""
    rows = []
    if base.kind == "textbook":
        for lv in levels:
            a = analyse(replace(base, level=lv))
            rows.append(_row(lv, a))
        return rows
    unit = replace(base, level=1.0)
    inst = instance_of(unit)
    g = A.Graph(inst)
    base_prizes = np.array(inst.prizes)
    al = A.steiner_all(g)
    tab = A.ExactTables(g) if inst.t <= C.N_EXACT else None
    for lv in levels:
        gl = g.with_prizes(base_prizes * lv)
        res, gw = A.gw_solutions(gl)
        sols = {"gw": gw["gw"], "gws": gw["gws"], "stp": A.make_solution("stp", gl, A.strong_prune(gl, al.edges)), "all": A.make_solution("all", gl, al.edges), "none": A.connect_none(gl)}
        if tab is not None:
            S_, _val = tab.best(gl.prizes)
            sols["exact"] = A.make_solution("exact", gl, tab.edges_of(S_))
        rows.append(_row(lv, None, sols=sols, dual=res.dual, total=float(gl.prizes.sum()), t=inst.t))
    return rows


def _row(level, a, sols=None, dual=None, total=None, t=None):
    if a is not None:
        sols, dual, total, t = a.sols, a.dual, a.inst.total_prize, a.t
    opt = "exact" if "exact" in sols else min(sols, key=lambda k: sols[k].objective)
    row = {"level": level, "opt_name": opt, "dual": dual, "total_prize": total, "t": t}
    for k, s in sols.items():
        row[k] = {"objective": s.objective, "net": s.net, "connected": len(s.connected), "cost": s.cost}
    row["opt"] = row[opt]
    return row


def nesting(base, seeds=C.FEAS_SEEDS, levels=C.LEVELS):
    """Ist die Menge der angeschlossenen Kunden im Optimum mit wachsendem Erlösniveau genestet? Je Instanz werden aufeinanderfolgende Stufen verglichen; eine Verletzung liegt vor, wenn KEIN optimaler
    Baum der höheren Stufe alle Kunden des optimalen Baums der niedrigeren Stufe enthält (Gleichstände: beste Teilmenge unter allen Obermengen gegen das Optimum, Toleranz 1e-6). Zusätzlich: fällt der
    gesammelte Erlös je einmal (darf nicht)."""
    viol, prize_drops, used, examples = 0, 0, 0, []
    for seed in seeds:
        inst = instance_of(replace(base, level=1.0, seed=seed))
        if inst.t > C.N_EXACT:
            continue
        g = A.Graph(inst)
        tab = A.ExactTables(g)
        unit = np.array(inst.prizes)
        prev, hit = None, False
        used += 1
        for lv in levels:
            pr = unit * lv
            vals = tab.values(pr)
            S_ = int(np.argmin(vals))
            if prev is not None:
                sup = np.array([m for m in range(1 << tab.k) if (m & prev[0]) == prev[0]])
                if vals[sup].min() > vals[S_] + 1e-6 and not hit:
                    hit = True
                    examples.append({"seed": seed, "from": prev[1], "to": lv})
                ps = tab.prize_of_sets(pr)
                if ps[S_] < ps[prev[0]] - 1e-6:
                    prize_drops += 1
            prev = (S_, lv)
        viol += hit
    return {"n_runs": used, "violated": viol, "violated_share": 100.0 * viol / used if used else float("nan"), "prize_drops": prize_drops, "examples": examples}


def exact_time_curve(base, ts=None):
    """Aufwand des exakten Wegs über die Kundenzahl (gleicher Plan, Seed und Einstellungen von `base`): Zustände (Teilmenge, Knoten) und gemessene Sekunden. Die Sekunden sind eine Messung (Rechner-abhängig)."""
    ts = ts if ts is not None else list(range(4, C.N_EXACT + 1))
    rows = []
    for t in ts:
        inst = instance_of(replace(base, t=t))
        g = A.Graph(inst)
        t0 = time.perf_counter()
        tab = A.ExactTables(g)
        S_, val = tab.best()
        rows.append({"t": t, "states": tab.states, "seconds": time.perf_counter() - t0, "value": val})
    return rows
