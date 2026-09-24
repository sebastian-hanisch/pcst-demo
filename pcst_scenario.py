"""Die Instanz dieser Demo: ein Stadtplan als gestörtes Gitter (wie in der Steiner-Baum-Demo). Kreuzungen sind die Knoten, Straßen zwischen Nachbarn die Kanten (Länge = euklidischer Abstand der leicht
verschobenen Kreuzungen); ein Anteil der Straßen ist gesperrt (der Plan bleibt zusammenhängend). Neu gegenüber der Steiner-Baum-Demo: das **Depot** muss im Baum liegen, die **Kunden** müssen es nicht -
jeder Kunde bringt einen **Erlös**, wenn er angeschlossen wird; jede andere Kreuzung ist ein möglicher **Steinerpunkt** (Erlös 0). Das **Erlösniveau** ist der Erlös eines Kunden in Straßenlängen
(ein Kunde mit Niveau 3 zahlt drei Nachbarstraßen); im Modus "gemischt" streut er zwischen 0.5 und 1.5 mal dem Niveau. Ein handgebautes Lehrbuchbeispiel (Gabel mit Ausläufer).

Knoten sind von 0 bis n - 1 durchnummeriert (Zeile für Zeile); Kanten (u, v, w) mit u < v, sortiert."""

from dataclasses import dataclass

import numpy as np

import pcst_constants as C
from pcst_unionfind import UnionFind


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray                 # (n, 2)
    edges: tuple                   # ((u, v, w), ...) sortiert
    depot: int
    customers: tuple               # sortierte Knotennummern
    prizes: tuple                  # Länge n, Erlös je Knoten (0 außer bei Kunden)
    side: int
    kind: str = "city"
    layout: str = "uniform"
    prize_mode: str = "equal"
    level: float = 0.0
    blocked: float = 0.0
    seed: int = 0
    blocked_edges: tuple = ()      # gesperrte Straßen (u, v), nur zur Anzeige

    @property
    def n(self):
        return len(self.xy)

    @property
    def m(self):
        return len(self.edges)

    @property
    def t(self):
        return len(self.customers)

    @property
    def terminals(self):
        return tuple(sorted((self.depot,) + tuple(self.customers)))

    @property
    def total_prize(self):
        return float(sum(self.prizes))


def _grid_edges(side):
    out = []
    for r in range(side):
        for c in range(side):
            v = r * side + c
            if c + 1 < side:
                out.append((v, v + 1))
            if r + 1 < side:
                out.append((v, v + side))
    return out


def _connected(n, pairs):
    uf = UnionFind(n, "full")
    for u, v in pairs:
        uf.union(u, v)
    return uf.components == 1


def _block(n, pairs, share, rng):
    """Sperrt `share` der Straßen in zufälliger Reihenfolge, überspringt eine Sperrung, wenn der Plan sonst zerfiele. Gibt (verbleibende, gesperrte) zurück."""
    target = int(round(share * len(pairs)))
    order = [int(i) for i in rng.permutation(len(pairs))]
    kept = set(range(len(pairs)))
    removed = []
    for i in order:
        if len(removed) >= target:
            break
        trial = [pairs[j] for j in kept if j != i]
        if _connected(n, trial):
            kept.discard(i)
            removed.append(i)
    return [pairs[j] for j in sorted(kept)], [pairs[j] for j in sorted(removed)]


def _pick_nodes(side, k, layout, rng):
    n = side * side
    if layout == "uniform":
        return sorted(int(x) for x in rng.choice(n, size=k, replace=False))
    g = 3 if k >= 6 else 2
    centers = rng.choice(n, size=g, replace=False)
    cxy = np.array([[c % side, c // side] for c in centers], dtype=float)
    pts = np.array([[v % side, v // side] for v in range(n)], dtype=float)
    chosen = []
    taken = set()
    for i in range(k):
        j = i % g
        d = np.hypot(pts[:, 0] - cxy[j, 0], pts[:, 1] - cxy[j, 1]) + rng.uniform(0.0, 0.01, size=n)
        for v in np.argsort(d, kind="stable"):
            if int(v) not in taken:
                taken.add(int(v))
                chosen.append(int(v))
                break
    return sorted(chosen)


def prizes_for(n, customers, prize_mode, level, seed, unit=C.SPACING):
    """Erlös je Knoten (Länge n): Niveau x Straßenlänge `unit`, im Modus "gemischt" mit einem Faktor aus [0.5, 1.5] je Kunde (eigener Zufallsstrom [seed, 909])."""
    if prize_mode not in C.PRIZE_MODES:
        raise ValueError(f"unbekannter Erlösmodus {prize_mode}")
    if not float(level) >= 0.0:
        raise ValueError("Erlösniveau negativ")
    out = np.zeros(n)
    factors = np.ones(len(customers)) if prize_mode == "equal" else np.random.default_rng([int(seed), 909]).uniform(0.5, 1.5, size=len(customers))
    for c, f in zip(customers, factors):
        out[c] = float(level) * unit * float(f)
    return tuple(float(x) for x in out)


def generate(side=C.DEFAULT_SIDE, t=C.DEFAULT_T, blocked=C.DEFAULT_BLOCKED, layout="uniform", prize_mode="equal", level=C.DEFAULT_LEVEL, seed=C.DEFAULT_SEED, jitter=C.JITTER):
    if layout not in C.LAYOUTS:
        raise ValueError(f"unbekanntes Layout {layout}")
    side, t = int(side), int(t)
    n = side * side
    if not 1 <= t <= n - 1:
        raise ValueError("Kundenzahl außerhalb")
    rng = np.random.default_rng([int(seed), 4242])
    xy = np.array([[c * C.SPACING, r * C.SPACING] for r in range(side) for c in range(side)], dtype=float)
    xy = xy + rng.uniform(-jitter, jitter, size=xy.shape) * C.SPACING
    kept, removed = _block(n, _grid_edges(side), float(blocked), rng)
    edges = tuple((u, v, float(np.hypot(*(xy[u] - xy[v])))) for u, v in kept)
    nodes = _pick_nodes(side, t + 1, layout, rng)
    depot, customers = nodes[0], tuple(nodes[1:])
    prizes = prizes_for(n, customers, prize_mode, level, seed)
    return Instance(xy, edges, depot, customers, prizes, side, "city", layout, prize_mode, float(level), float(blocked), int(seed), tuple(removed))


# --- Handgebautes Lehrbuchbeispiel ----------------------------------------------------------------------------------------------------------------


def textbook_instance(level=3.0, small=C.TEXTBOOK_SMALL_PRIZE):
    """Gabel mit Ausläufer, alle Straßen Länge 1 (Knoten 0 Depot D, 1 a, 2 b, 3 Gabel F, 4 Kunde A, 5 Kunde B, 6 Kunde C): D - a - b - F ist ein Stamm der Länge 3, von F gehen A und B ab (je 1), C hängt
    mit Länge 1 direkt am Depot; zusätzlich eine Straße A - B der Länge 2, die nie gebraucht wird. A und B zahlen je `level` (Voreinstellung 3), C zahlt `small` (0.6). Einzeln lohnt sich weder A noch B
    (Stamm 3 + Ast 1 = 4 > 3), gemeinsam schon (3 + 1 + 1 = 5 < 6); C lohnt sich nie (0.6 < 1). Optimum bei Niveau 3: A und B anschließen, Kosten 5, Zielwert 5 + 0.6 = 5.6."""
    xy = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 1], [4, -1], [0, 1]], dtype=float)
    edges = ((0, 1, 1.0), (0, 6, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 2.0))
    prizes = (0.0, 0.0, 0.0, 0.0, float(level), float(level), float(small))
    return Instance(xy, edges, 0, (4, 5, 6), prizes, 0, "textbook", "uniform", "equal", float(level))
