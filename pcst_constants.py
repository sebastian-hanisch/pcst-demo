"""Konstanten der Prize-Collecting-Steiner-Demo: Instanz-Geometrie, Regler, gemessene Werte, Presets."""
SPACING = 10.0
JITTER = 0.15
SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 5, 14, 8
T_MIN, T_MAX, DEFAULT_T = 3, 30, 8
BLOCKED_OPTIONS = (0.0, 0.1, 0.2, 0.3, 0.4)
DEFAULT_BLOCKED = 0.1
SEED_MAX = 999999
DEFAULT_SEED = 10
KINDS = ("city", "textbook")
KIND_LABELS = {"city": "Stadtplan (Gitter)", "textbook": "Lehrbuchbeispiel (Gabel mit Ausläufer)"}
LAYOUTS = ("uniform", "clusters")
LAYOUT_LABELS = {"uniform": "gleichverteilt", "clusters": "in Gruppen"}
PRIZE_MODES = ("equal", "mixed")
PRIZE_LABELS = {"equal": "gleich hoch", "mixed": "gemischt (0.5-1.5 x)"}
LEVELS = tuple(0.5 + 0.25 * i for i in range(15)) + (4.5, 5.0, 5.5, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
DEFAULT_LEVEL = 2.5
TEXTBOOK_SMALL_PRIZE = 0.6
SWEEP_SEEDS = tuple(range(100000, 100005))
FEAS_SEEDS = tuple(range(200000, 200050))
N_EXACT = 12
TREE_OPTIONS = ("best", "exact", "gw", "gws", "gwr", "stp", "ls", "all", "none")
TREE_LABELS = {"best": "Bester Fund", "exact": "Exakt", "gw": "Goemans-Williamson (GW-Beschneiden)", "gws": "Goemans-Williamson (starkes Beschneiden)",
               "gwr": "Goemans-Williamson (ohne Beschneiden)", "stp": "Erst Steinerbaum, dann kürzen", "ls": "Lokalsuche", "all": "Alle anschließen", "none": "Nichts anschließen"}
DEFAULT_TREE = "best"

_BASE = {"kind": "city", "side": 8, "t": 8, "blocked": 0.1, "layout": "uniform", "prize_mode": "equal", "level": 2.5, "seed": 10, "tree": "best"}
PRESETS = {
    "Standardfall (Voreinstellung)": dict(_BASE),
    "Lehrbuchbeispiel (Gabel)": {**_BASE, "kind": "textbook", "level": 3.0},
    "Niedriges Erlösniveau": {**_BASE, "level": 1.5, "seed": 37, "tree": "gw"},
    "Hohes Erlösniveau": {**_BASE, "level": 6.0, "seed": 0, "tree": "gws"},
    "Gruppierte Kunden": {**_BASE, "layout": "clusters", "t": 9, "level": 1.5, "seed": 47},
    "GW weit über dem Optimum": {**_BASE, "seed": 1, "tree": "gws"},
    "Erst Steiner verliert gegen GW": {**_BASE, "seed": 38, "tree": "stp"},
    "Lokalsuche bleibt hängen": {**_BASE, "seed": 60, "tree": "ls"},
    "Nestung verletzt": {**_BASE, "prize_mode": "mixed", "level": 3.25, "seed": 12},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Plan 8 x 8 (64 Kreuzungen, 11 gesperrte Straßen), 8 Kunden mit je 2.5 Straßenlängen Erlös (zusammen 200), Seed 10: das Optimum schließt 5 von 8 Kunden an (Trassenkosten 90.15, Netto-Erlös 34.85, Zielwert 165.15). Alle anschließen kostet 177.51 (+7.49 %), nichts anschließen 200 (+21.10 %): Wert der Auswahl 6.96 %. Erst Steinerbaum, dann kürzen und die Lokalsuche finden das Optimum; GW mit starkem Beschneiden liegt +1.07 % darüber, mit GW-Beschneiden +8.56 % (alle 8 Kunden angeschlossen).",
    "Lehrbuchbeispiel (Gabel)": "Gabel mit Ausläufer auf 7 Kreuzungen: ein Stamm der Länge 3, dann zwei Äste der Länge 1 zu A und B (je Erlös 3), Kunde C am Depot (Erlös 0.6, Straße 1). Einzeln lohnt sich weder A noch B (3 + 1 = 4 > 3), gemeinsam schon (3 + 1 + 1 = 5 < 6); C lohnt sich nie (0.6 < 1). Optimum: A und B anschließen, Kosten 5, Zielwert 5.6, Netto-Erlös 1. Alle anschließen kostet 6, nichts anschließen 6.6; die Untergrenze von GW ist 5.6 - hier genau das Optimum. Bei Niveau 2.5 sind \"nichts\" und \"A und B\" gleich gut (5.6).",
    "Niedriges Erlösniveau": "Niveau 1.5, Seed 37: das Optimum schließt nur 3 von 8 Kunden an (Zielwert 105.60, Wert der Auswahl 12.00 %); alle anschließen wäre +38.19 % schlechter. GW mit GW-Beschneiden schließt 7 Kunden an und liegt +34.35 % über dem Optimum; starkes Beschneiden (2 Kunden) nur +4.04 %.",
    "Hohes Erlösniveau": "Niveau 6, Seed 0: es lohnt sich jeder Kunde, das Optimum ist der Steinerbaum über alle 8 (Kosten 196.73, gleich \"alle anschließen\"; nichts anschließen wäre +143.99 % schlechter). GW liegt +12.88 % darüber - hier verliert es als Steinerbaum-Heuristik; erst Steinerbaum und Lokalsuche finden das Optimum. Untergrenze von GW: 64 % des Optimums.",
    "Gruppierte Kunden": "9 Kunden in Gruppen, Niveau 1.5, Seed 47: das Optimum schließt 5 von 9 an (Zielwert 108.62); alle anschließen +38.17 %, nichts +24.29 %: Wert der Auswahl 19.54 %. GW mit starkem Beschneiden, erst Steiner und die Lokalsuche finden es, GW mit GW-Beschneiden liegt +7.16 % darüber.",
    "GW weit über dem Optimum": "Seed 1: GW mit starkem Beschneiden endet bei 175.03 (+14.11 % über dem Optimum 153.39), mit GW-Beschneiden bei 182.13 (+18.74 %), der Netto-Erlös fällt dabei von 46.61 auf 24.97 (−46.4 %). Erst Steinerbaum und Lokalsuche liegen bei 153.57 (+0.12 %). Untergrenze von GW: 107.91 (70 % des Optimums).",
    "Erst Steiner verliert gegen GW": "Seed 38: \"erst Steinerbaum, dann kürzen\" endet bei 134.97 (+5.77 % über dem Optimum 127.60), GW (beide Beschneidungen) und die Lokalsuche finden das Optimum. Der Netto-Erlös fällt bei \"erst Steiner\" von 72.40 auf 65.03 (−10.2 %).",
    "Lokalsuche bleibt hängen": "Seed 60: die Lokalsuche endet bei 174.38 (+5.68 % über dem Optimum 165.01), ebenso \"erst Steiner\"; GW stark +9.13 %. Das Optimum schließt 7 von 8 Kunden an.",
    "Nestung verletzt": "Gemischte Erlöse, Seed 12: bei Niveau 3 schließt das Optimum 6 Kunden an (darunter Kunde 58), bei 3.25 sind es 7 - aber ohne Kunde 58: die Kunden 12 und 22 kommen dazu, 58 fällt heraus, obwohl sein Erlös steigt. Die Anschlussmenge ist also nicht genestet. Ziehen Sie das Niveau von 3 auf 3.25.",
}
# Beobachtete Spannweite der Kennzahl (MEDIAN über die 5 festen Instanzen Seeds 100000-100004) je Preset, mit Sicherheitsabstand: (Kennzahl, untere, obere Grenze).
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": ("connected_share", 75.0, 100.0),
    "Niedriges Erlösniveau": ("connected_share", 0.0, 12.5),
    "Hohes Erlösniveau": ("connected_share", 99.0, 100.0),
    "Gruppierte Kunden": ("connected_share", 22.0, 45.0),
    "GW weit über dem Optimum": ("connected_share", 75.0, 100.0),
    "Erst Steiner verliert gegen GW": ("connected_share", 75.0, 100.0),
    "Lokalsuche bleibt hängen": ("connected_share", 75.0, 100.0),
    "Nestung verletzt": ("connected_share", 75.0, 100.0),
}
