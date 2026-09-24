"""Prize-Collecting Steiner-Baum (PCST) in Graphen: gegeben ein Graph mit Kantenlängen, ein **Depot** (muss im Baum liegen) und **Kunden** mit Erlösen. Gesucht ist ein Baum durch das Depot, der die Summe aus
Trassenkosten und den Erlösen der NICHT angeschlossenen Kunden minimiert (Zielwert). Der **Netto-Erlös** (Erlöse der angeschlossenen Kunden minus Kosten) ist dazu im Optimum gleichwertig - Gesamterlös minus Zielwert -,
in der Näherungsgüte aber nicht. Nicht angeschlossene Kunden sind erlaubt, Zwischenknoten (Steinerpunkte) auch.

**Kernsatz:** jede Lösung ist ein Baum auf einer Knotenmenge X mit Depot, ihr Minimum der MST des induzierten Teilgraphen G[X]: OPT = min über X von MST(G[X]) + Erlöse der Kunden außerhalb von X.

Verfahren: **Goemans-Williamson** (`gw_primal_dual`, 1995: Primal-Dual-Wachstum, aktive Komponenten wachsen bis ihr Erlös aufgebraucht ist; Untergrenze `dual`; Beschneiden nach GW oder **stark** nach Johnson-Minkoff-Phillips,
`strong_prune`), **erst Steinerbaum über alle Kunden, dann optimal kürzen** (`steiner_then_prune`, Bausteine Kou-Markowsky-Berman und Takahashi-Matsuyama aus der Steiner-Baum-Demo), eine **Lokalsuche** über
Knotenmengen (`local_search`) und **exakt** nach Dreyfus-Wagner (`pcst_exact`: dynamische Programmierung über Teilmengen der Kunden, O(3^t * n)). Alle deterministisch (Schlüssel: Zeit, Art, Index)."""

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from pcst_unionfind import UnionFind

EPS = 1e-9


class Graph:
    """Der Stadtplan als Graph: Kantenlängen, Nachbarlisten, Abstände und Nachfolger aller Knotenpaare (Floyd-Warshall, numpy); dazu Depot, Kunden und Erlöse."""

    def __init__(self, inst):
        self.n = inst.n
        self.depot = int(inst.depot)
        self.customers = tuple(inst.customers)
        self.prizes = np.array(inst.prizes, dtype=float)
        self.terminals = tuple(sorted((self.depot,) + self.customers))
        self.w = {(u, v): float(w) for u, v, w in inst.edges}
        self.edges = sorted(self.w)
        self.adj = [[] for _ in range(self.n)]
        for (u, v), w in self.w.items():
            self.adj[u].append(v)
            self.adj[v].append(u)
        for a in self.adj:
            a.sort()
        self.dist, self.nxt = all_pairs(self.n, self.w)

    def path(self, u, v):
        """Knotenfolge eines kürzesten Wegs u -> v."""
        out = [u]
        while u != v:
            u = int(self.nxt[u, v])
            out.append(u)
        return out

    def path_edges(self, u, v):
        p = self.path(u, v)
        return [(min(a, b), max(a, b)) for a, b in zip(p, p[1:])]

    def cost(self, edges):
        return math.fsum(self.w[e] for e in edges)

    def with_prizes(self, prizes):
        """Derselbe Plan mit anderen Erlösen (Abstände und Nachbarlisten werden geteilt, nicht neu berechnet)."""
        other = copy.copy(self)
        other.prizes = np.array(prizes, dtype=float)
        return other


def all_pairs(n, w):
    """Abstände und Nachfolger-Matrix aller Knotenpaare (Floyd-Warshall); `w` = {(u, v): Länge} mit u < v. nxt[i, j] = erster Knoten auf dem kürzesten Weg i -> j (-1: unerreichbar)."""
    dist = np.full((n, n), np.inf)
    nxt = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(dist, 0.0)
    for i in range(n):
        nxt[i, i] = i
    for (u, v), x in w.items():
        if x < dist[u, v]:
            dist[u, v] = dist[v, u] = x
            nxt[u, v], nxt[v, u] = v, u
    for k in range(n):
        via = dist[:, k:k + 1] + dist[k:k + 1, :]
        better = via < dist - 1e-12
        if better.any():
            dist = np.where(better, via, dist)
            nxt = np.where(better, nxt[:, k:k + 1], nxt)
    return dist, nxt


# --- Zielwert, Lösungen ---------------------------------------------------------------------------------------------------------------------------


def norm_edges(edges):
    return sorted({(min(u, v), max(u, v)) for u, v in edges})


def tree_nodes(g, edges):
    """Knoten des Baums; das Depot gehört immer dazu (leerer Baum = nur das Depot)."""
    return {g.depot} | {x for e in edges for x in e}


def objective(g, edges):
    """Zielwert = Trassenkosten + Erlöse der nicht angeschlossenen Kunden."""
    nodes = tree_nodes(g, edges)
    return g.cost(edges) + math.fsum(g.prizes[c] for c in g.customers if c not in nodes)


def net_worth(g, edges):
    """Netto-Erlös = Erlöse der angeschlossenen Kunden - Trassenkosten."""
    nodes = tree_nodes(g, edges)
    return math.fsum(g.prizes[c] for c in g.customers if c in nodes) - g.cost(edges)


@dataclass
class Solution:
    method: str
    edges: list                                    # (u, v) mit u < v
    cost: float                                    # Trassenkosten
    objective: float                               # Kosten + Erlöse der nicht angeschlossenen Kunden
    net: float                                     # Netto-Erlös
    connected: list                                # angeschlossene Kunden
    steiner: list                                  # Steinerpunkte (Baumknoten ohne Erlös, außer Depot)
    detail: dict = field(default_factory=dict)


def make_solution(method, g, edges, detail=None):
    edges = norm_edges(edges)
    nodes = tree_nodes(g, edges)
    cs = set(g.customers)
    return Solution(method, edges, g.cost(edges), objective(g, edges), net_worth(g, edges), [c for c in g.customers if c in nodes],
                    sorted(nodes - cs - {g.depot}), dict(detail or {}))


def is_pcst_solution(g, edges):
    """Alle Kanten aus G, keine doppelten, azyklisch, zusammenhängend und (bei mindestens einer Kante) durch das Depot; die leere Kantenmenge ist der leere Baum."""
    edges = list(edges)
    if len(set(edges)) != len(edges) or any(e not in g.w for e in edges):
        return False
    if not edges:
        return True
    uf = UnionFind(g.n, "full")
    if not all(uf.union(u, v) for u, v in edges):
        return False
    nodes = {x for e in edges for x in e}
    return g.depot in nodes and len({uf.find(x) for x in nodes}) == 1


def kruskal_edges(g, edge_list):
    """Kruskal auf den gegebenen Kanten (Schlüssel (Länge, u, v)); gibt den aufspannenden Wald zurück."""
    uf = UnionFind(g.n, "full")
    return [e for e in sorted(edge_list, key=lambda e: (g.w[e], e)) if uf.union(e[0], e[1])]


def prune(g, edges, keep=None):
    """Blätter, die nicht in `keep` liegen (Standard: Depot und alle Kunden), wiederholt entfernen."""
    edges = set(edges)
    keep = set(g.terminals) if keep is None else set(keep)
    while True:
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        leaves = {x for x, d in deg.items() if d == 1 and x not in keep}
        if not leaves:
            return sorted(edges)
        edges = {e for e in edges if e[0] not in leaves and e[1] not in leaves}


def branch_points(g, edges):
    """Nicht-Kunden-Knoten (ohne Depot) mit Grad >= 3 im Baum: die echten Verzweigungen."""
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    return sorted(x for x, d in deg.items() if d >= 3 and x not in g.customers and x != g.depot)


# --- Starkes Beschneiden (Johnson-Minkoff-Phillips) -----------------------------------------------------------------------------------------------


def strong_prune(g, edges):
    """Optimales Beschneiden eines Baums durch das Depot: bottom-up gain(v) = Erlös(v) + Summe über Kinder k von max(0, gain(k) - Länge(v, k)); ein Teilbaum bleibt nur, wenn er mehr einbringt, als seine
    Anschlusskante kostet (Gleichstand: weg). Maximiert den Netto-Erlös über alle Teilbäume des Baums, die das Depot enthalten."""
    edges = norm_edges(edges)
    if not edges:
        return []
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    if g.depot not in adj:
        raise ValueError("Baum enthält das Depot nicht")
    parent = {g.depot: None}
    order = []
    stack = [g.depot]
    while stack:
        x = stack.pop()
        order.append(x)
        for y in sorted(adj[x]):
            if y not in parent:
                parent[y] = x
                stack.append(y)
    gain, kids = {}, {}
    for x in reversed(order):
        val = float(g.prizes[x])
        kids[x] = []
        for y in adj[x]:
            if parent.get(y) == x:
                s = gain[y] - g.w[(min(x, y), max(x, y))]
                if s > EPS:
                    val += s
                    kids[x].append(y)
        gain[x] = val
    kept, todo = [], [g.depot]
    while todo:
        x = todo.pop()
        for y in kids[x]:
            kept.append((min(x, y), max(x, y)))
            todo.append(y)
    return sorted(kept)


# --- Goemans-Williamson ---------------------------------------------------------------------------------------------------------------------------


@dataclass
class GWResult:
    forest: list                                   # feste Kanten in der Reihenfolge ihres Festwerdens
    events: list                                   # je Ereignis: time, kind ("merge" | "dead"), edge / members, comp (Komponente je Knoten), active (0/1 je Knoten), d (Last je Knoten), dual, n_active
    dual: float                                    # Summe aller Moat-Variablen y_S = Integral der Zahl aktiver Komponenten: Untergrenze des Zielwerts
    marks: list                                    # je Knoten Nummer der größten toten Menge, die ihn enthält (-1: nie in einer toten Menge)
    dead_sets: list                                # tote Mengen (frozenset) in der Reihenfolge ihres Sterbens
    root_edges: list                               # Kanten der Komponente des Depots am Ende (der Rohbaum)
    final_comp: list                               # Komponente je Knoten am Ende


def gw_primal_dual(g):
    """Wachstumsphase von Goemans-Williamson (gewurzelt): jede Komponente ohne Depot ist aktiv, solange ihr Restpotenzial Erlös - Moat-Summe positiv ist; aktive Komponenten laden alle ihre Knoten mit der Zeit auf
    (d_v). Ereignis 1: eine Kante (u, v) zwischen zwei Komponenten wird fest, wenn d_u + d_v = Länge (mit Rate 2 bei zwei aktiven Seiten, 1 bei einer); Ereignis 2: eine aktive Komponente stirbt, wenn ihr
    Restpotenzial 0 ist. Beim Verschmelzen addieren sich die Restpotenziale; enthält das Ergebnis das Depot, ist es nie mehr aktiv. Gleichstand: erst sterben, dann Kante; Kanten nach Index."""
    n = g.n
    eu = np.array([e[0] for e in g.edges], dtype=int)
    ev = np.array([e[1] for e in g.edges], dtype=int)
    ew = np.array([g.w[e] for e in g.edges], dtype=float)
    depot = g.depot
    cid = np.arange(n)
    d = np.zeros(n)
    pot = {v: float(g.prizes[v]) for v in range(n)}
    active = {v: bool(v != depot and g.prizes[v] > EPS) for v in range(n)}
    pot[depot] = math.inf
    members = {v: [v] for v in range(n)}
    marks = [-1] * n
    dead_sets = []
    for v in range(n):
        if v != depot and not active[v]:
            dead_sets.append(frozenset([v]))
            marks[v] = len(dead_sets) - 1
    forest, events = [], []
    time = 0.0
    dual = 0.0
    while True:
        act_ids = sorted(c for c, a in active.items() if a)
        if not act_ids:
            break
        act_node = np.array([1.0 if active[int(cid[v])] else 0.0 for v in range(n)])
        diff = cid[eu] != cid[ev]
        rate = act_node[eu] + act_node[ev]
        ok = diff & (rate > 0)
        cand = []
        if ok.any():
            dt_e = np.full(len(eu), math.inf)
            dt_e[ok] = np.maximum((ew[ok] - d[eu][ok] - d[ev][ok]) / rate[ok], 0.0)
            i = int(np.argmin(dt_e))
            cand.append((float(dt_e[i]), 1, i))
        for c in act_ids:
            cand.append((max(pot[c], 0.0), 0, c))
        mn = min(x[0] for x in cand)
        dt, kind, idx = min((x for x in cand if x[0] <= mn + 1e-12), key=lambda x: (x[1], x[2]))
        if kind == 1:
            i = int(np.flatnonzero(dt_e <= dt_e[idx] + 1e-12)[0])
            idx = i
        for v in range(n):
            if act_node[v]:
                d[v] += dt
        for c in act_ids:
            pot[c] = max(pot[c] - dt, 0.0)
        dual += len(act_ids) * dt
        time += dt
        if kind == 0:
            c = idx
            active[c] = False
            pot[c] = 0.0
            dead_sets.append(frozenset(members[c]))
            for v in members[c]:
                marks[v] = len(dead_sets) - 1
            events.append({"time": time, "kind": "dead", "members": sorted(members[c])})
        else:
            u, v = int(eu[idx]), int(ev[idx])
            a, b = int(cid[u]), int(cid[v])
            if b == int(cid[depot]):
                a, b = b, a
            members[a] += members[b]
            for x in members[b]:
                cid[x] = a
            pot[a] = pot[a] + pot[b]
            active[a] = bool(a != int(cid[depot]) and pot[a] > EPS)
            if a == int(cid[depot]):
                active[a] = False
            del members[b], pot[b], active[b]
            forest.append((u, v))
            events.append({"time": time, "kind": "merge", "edge": (u, v)})
        act_after = np.array([1 if active[int(cid[v])] else 0 for v in range(n)])
        events[-1].update({"comp": cid.copy().tolist(), "active": act_after.tolist(), "d": d.copy().tolist(), "dual": dual, "n_active": sum(1 for c in active if active[c])})
    root = int(cid[depot])
    root_edges = norm_edges(e for e in forest if int(cid[e[0]]) == root)
    return GWResult(forest, events, dual, marks, dead_sets, root_edges, cid.tolist())


def _subtree_spanning(edges, keep):
    """Kleinster Teilbaum des Baums `edges`, der alle Knoten aus `keep` enthält (Blätter außerhalb von `keep` wiederholt entfernen)."""
    edges = set(edges)
    while True:
        deg = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        leaves = {x for x, dg in deg.items() if dg == 1 and x not in keep}
        if not leaves:
            return sorted(edges)
        edges = {e for e in edges if e[0] not in leaves and e[1] not in leaves}


def gw_prune(g, res):
    """GW-Beschneiden (Goemans & Williamson 1995): so viele Kanten des Rohbaums entfernen wie möglich, ohne dass (i) ein Knoten, der nie in einer toten Menge lag, vom Depot getrennt wird und (ii) eine tote
    Menge teilweise angeschlossen bleibt (ist ein Knoten mit Markierung C angeschlossen, sind es alle Knoten mit derselben Markierung). Ergebnis: der eindeutige kleinste Teilbaum, der das Depot, alle unmarkierten
    Knoten und - bis zum Fixpunkt - alle Knoten der angetroffenen Markierungsklassen enthält. Zusätzlich werden übrig gebliebene Nicht-Kunden-Blätter abgeschnitten (nur billiger, gleiche Kundenmenge)."""
    tree = res.root_edges
    nodes = {x for e in tree for x in e} | {g.depot}
    keep = {g.depot} | {v for v in nodes if res.marks[v] == -1}
    while True:
        sub = _subtree_spanning(tree, keep)
        sub_nodes = {x for e in sub for x in e} | {g.depot}
        classes = {res.marks[x] for x in sub_nodes if res.marks[x] != -1}
        add = {v for v in nodes if res.marks[v] in classes}
        if add <= keep:
            return prune(g, sub)
        keep |= add


def gw_solutions(g):
    """Die drei Ergebnisse der Goemans-Williamson-Läufe: roh (Komponente des Depots), GW-Beschneiden, starkes Beschneiden."""
    res = gw_primal_dual(g)
    det = {"events": len(res.events), "dual": res.dual}
    return res, {
        "gwr": make_solution("gwr", g, res.root_edges, det),
        "gw": make_solution("gw", g, gw_prune(g, res), det),
        "gws": make_solution("gws", g, strong_prune(g, res.root_edges), det),
    }


# --- Steinerbaum über alle Kunden (Bausteine aus der Steiner-Baum-Demo) und "erst bauen, dann kürzen" ---------------------------------------------


def closure_mst(g):
    """MST über die Terminals (Depot und Kunden) mit den Abständen des Graphen (Metrik-Abschluss): Kanten (a, b, Abstand)."""
    T = g.terminals
    pairs = sorted((float(g.dist[a, b]), a, b) for i, a in enumerate(T) for b in T[i + 1:])
    uf = UnionFind(g.n, "full")
    return [(a, b, d) for d, a, b in pairs if uf.union(a, b)]


def kmb(g):
    """Kou-Markowsky-Berman über alle Terminals: Metrik-Abschluss-MST, Kanten zu kürzesten Wegen expandieren, MST des Teilgraphen, Blätter beschneiden."""
    if len(g.terminals) <= 1:
        return make_solution("kmb", g, [])
    expanded = set()
    for a, b, _d in closure_mst(g):
        expanded.update(g.path_edges(a, b))
    return make_solution("kmb", g, prune(g, kruskal_edges(g, expanded)))


def takahashi_matsuyama(g, root=None):
    """Der Baum wächst von `root` (Standard: Depot); jeweils das dem Baum nächste Terminal über den kürzesten Weg (Gleichstand: kleinster Abstand, dann kleinstes Terminal, dann kleinster Baumknoten)."""
    T = list(g.terminals)
    root = g.depot if root is None else root
    nodes = {root}
    edges = set()
    rest = [x for x in T if x != root]
    while rest:
        best = None
        for r in rest:
            for v in sorted(nodes):
                key = (float(g.dist[v, r]), r, v)
                if best is None or key < best:
                    best = key
        _d, r, v = best
        edges.update(g.path_edges(v, r))
        nodes.update(g.path(v, r))
        rest.remove(r)
    return make_solution("tm", g, edges, {"root": root})


def induced_steiner_tree(g, nodes):
    """Der beschnittene MST des von `nodes` induzierten Teilgraphen (Blätter bleiben nur, wenn sie Terminals sind); None, wenn er nicht zusammenhängt."""
    nodes = set(nodes)
    edges = [e for e in g.edges if e[0] in nodes and e[1] in nodes]
    mst = kruskal_edges(g, edges)
    if len(mst) != len(nodes) - 1:
        return None
    return prune(g, mst)


def steiner_local_search(g, start_nodes):
    """Lokalsuche über Steinerpunkte für den Steinerbaum über alle Terminals: einen Knoten einfügen oder entfernen, bewertet mit dem MST des induzierten Teilgraphen; streng fallende Kosten."""
    ts = set(g.terminals)
    X = set(start_nodes) - ts
    cur = induced_steiner_tree(g, ts | X)
    if cur is None:
        raise ValueError("Startmenge verbindet die Terminals nicht")
    cost = g.cost(cur)
    while True:
        base = ts | X
        cand = [("add", v) for v in range(g.n) if v not in base and any(x in base for x in g.adj[v])] + [("del", x) for x in sorted(X)]
        best = None
        for kind, v in cand:
            Y = X | {v} if kind == "add" else X - {v}
            tree = induced_steiner_tree(g, ts | Y)
            if tree is None:
                continue
            c = g.cost(tree)
            if c < cost - EPS and (best is None or c < best[0] - EPS):
                best = (c, Y, tree)
        if best is None:
            return make_solution("stp_ls", g, cur)
        cost, X, cur = best
        X = {x for e in cur for x in e} - ts


def steiner_all(g):
    """Der beste Steinerbaum über Depot und ALLE Kunden, den die Heuristiken finden (KMB, Takahashi-Matsuyama mit jedem Terminal als Wurzel, Lokalsuche darauf): die Grundlage von "alle anschließen"."""
    if len(g.terminals) <= 1:
        return make_solution("all", g, [])
    cands = [kmb(g)] + [takahashi_matsuyama(g, r) for r in g.terminals]
    best = min(cands, key=lambda s: (s.cost, s.method))
    ls = steiner_local_search(g, {x for e in best.edges for x in e})
    best = ls if ls.cost < best.cost - EPS else best
    return make_solution("all", g, best.edges)


def steiner_then_prune(g, base=None):
    """"Erst bauen, dann kürzen": der Steinerbaum über alle Kunden, danach optimal beschnitten (`strong_prune`)."""
    base = steiner_all(g) if base is None else base
    return make_solution("stp", g, strong_prune(g, base.edges))


def connect_none(g):
    return make_solution("none", g, [])


# --- Lokalsuche über Knotenmengen -----------------------------------------------------------------------------------------------------------------


def induced_pcst_tree(g, nodes):
    """Der MST des von `nodes` (mit Depot) induzierten Teilgraphen, Nicht-Kunden-Blätter beschnitten; None, wenn der Teilgraph nicht zusammenhängt."""
    nodes = set(nodes) | {g.depot}
    edges = [e for e in g.edges if e[0] in nodes and e[1] in nodes]
    mst = kruskal_edges(g, edges)
    if len(mst) != len(nodes) - 1:
        return None
    return prune(g, mst)


def local_search(g, start_edges):
    """Lokalsuche über Knotenmengen X mit Depot (Start: Knoten des Startbaums); Wert(X) = Zielwert des beschnittenen MST von G[X]. Beste Verbesserung je Runde durch Einfügen eines Nachbarknotens (Kunde oder
    Steinerpunkt) oder Entfernen eines Knotens außer dem Depot, streng fallend; Ende im lokalen Optimum. `detail['rounds']`: Zahl der Verbesserungen."""
    X = {x for e in start_edges for x in e} | {g.depot}
    cur = induced_pcst_tree(g, X)
    if cur is None:
        raise ValueError("Startmenge nicht zusammenhängend")
    val = objective(g, cur)
    start_val = val
    rounds = 0
    while True:
        X = {x for e in cur for x in e} | {g.depot}
        cand = [("add", v) for v in range(g.n) if v not in X and any(x in X for x in g.adj[v])] + [("del", x) for x in sorted(X) if x != g.depot]
        best = None
        for kind, v in cand:
            Y = X | {v} if kind == "add" else X - {v}
            tree = induced_pcst_tree(g, Y)
            if tree is None:
                continue
            c = objective(g, tree)
            if c < val - EPS and (best is None or c < best[0] - EPS):
                best = (c, tree)
        if best is None:
            return make_solution("ls", g, cur, {"rounds": rounds, "start": start_val})
        val, cur = best
        rounds += 1


# --- Exakt: Dreyfus-Wagner über die Kunden --------------------------------------------------------------------------------------------------------


@dataclass
class ExactResult:
    solution: object
    states: int                                    # Zahl der (Teilmenge, Knoten)-Zustände
    value: float                                   # Zielwert laut Tabelle
    tables: object = None                          # `ExactTables` (für weitere Erlösniveaus ohne neue Rechnung)


class ExactTables:
    """Dreyfus-Wagner-Tabelle über alle t Kunden (Teilmengen-Bits, das Depot als Wurzel): dp[S][v] = kürzester Baum, der die Kunden in S und den Knoten v verbindet;
    dp[S][v] = min_u (min_{A + B = S} dp[A][u] + dp[B][u]) + dist(u, v). O(3^t * n). Die Tabelle hängt nicht von den Erlösen ab - nur `best` tut es."""

    def __init__(self, g):
        self.g = g
        self.cs = list(g.customers)
        k = len(self.cs)
        n = g.n
        self.k = k
        self.states = int(((1 << k) - 1) * n) if k else 0
        full = (1 << k) - 1
        self.dp = np.full((1 << k, n), np.inf)
        self.split = np.zeros((1 << k, n), dtype=int)
        self.arg = np.zeros((1 << k, n), dtype=int)
        for i in range(k):
            self.dp[1 << i] = g.dist[:, self.cs[i]]
        for S in range(1, full + 1):
            if S & (S - 1) == 0:
                continue
            low = S & -S
            rest = S ^ low
            best = np.full(n, np.inf)
            bestA = np.zeros(n, dtype=int)
            s = rest
            while True:
                if s != rest:
                    A = low | s
                    cand = self.dp[A] + self.dp[S ^ A]
                    better = cand < best - 1e-12
                    best = np.where(better, cand, best)
                    bestA = np.where(better, A, bestA)
                if s == 0:
                    break
                s = (s - 1) & rest
            M = best[:, None] + g.dist
            self.arg[S] = np.argmin(M, axis=0)
            self.dp[S] = M[self.arg[S], np.arange(n)]
            self.split[S] = bestA
        self.tree_cost = self.dp[:, g.depot].copy()
        self.tree_cost[0] = 0.0

    def prize_of_sets(self, prizes):
        """Erlös jeder Kundenteilmenge (Array der Länge 2^t)."""
        out = np.zeros(1 << self.k)
        for S in range(1, 1 << self.k):
            low = S & -S
            out[S] = out[S ^ low] + prizes[self.cs[low.bit_length() - 1]]
        return out

    def values(self, prizes):
        """Zielwert je Teilmenge S: Baum für S + Erlöse der Kunden außerhalb von S."""
        ps = self.prize_of_sets(prizes)
        return self.tree_cost + (ps[-1] - ps)

    def edges_of(self, S):
        g = self.g
        if S == 0:
            return []
        edges = set()

        def build(S, v):
            if S & (S - 1) == 0:
                edges.update(g.path_edges(v, self.cs[S.bit_length() - 1]))
                return
            u = int(self.arg[S][v])
            edges.update(g.path_edges(v, u))
            A = int(self.split[S][u])
            build(A, u)
            build(S ^ A, u)

        build(S, g.depot)
        return prune(g, kruskal_edges(g, edges))

    def best(self, prizes=None):
        """Beste Teilmenge S und ihr Wert für die Erlöse `prizes` (Standard: die der Instanz); Gleichstand: kleinste Teilmenge (Maske)."""
        vals = self.values(self.g.prizes if prizes is None else prizes)
        S = int(np.argmin(vals))
        return S, float(vals[S])


def pcst_exact(g):
    """Exakter Zielwert: OPT = min über Kundenteilmengen S von dp[S][Depot] + Erlöse der Kunden außerhalb von S (S leer: der leere Baum); die Lösung wird aus der Tabelle zurückverfolgt."""
    if not g.customers:
        return ExactResult(make_solution("exact", g, []), 0, 0.0)
    tab = ExactTables(g)
    S, val = tab.best()
    return ExactResult(make_solution("exact", g, tab.edges_of(S), {"dp_value": val, "set": S}), tab.states, val, tab)
