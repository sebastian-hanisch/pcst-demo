# Prize-Collecting Steiner-Baum – Goemans-Williamson, Beschneiden, Lokalsuche, exakt – Streamlit-Demo

Neuntes Stück der **Spannbaum-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning". Bisher musste **jeder Kunde** angeschlossen werden. Bringt ein Anschluss aber einen **Erlös**, lohnt sich ein Kunde am Ende einer langen Trasse vielleicht nicht – und zwei Kunden, die einzeln zu weit weg sind, lohnen sich zusammen, weil sie sich einen Stamm teilen. Gegeben ist ein Stadtplan (Kreuzungen, Straßen mit Länge, ein Teil gesperrt), ein **Depot**, das im Baum liegen muss, und **Kunden mit Erlös**; gesucht ist der Baum, der die Summe aus **Trassenkosten und den Erlösen der nicht angeschlossenen Kunden** minimiert (der **Zielwert**; Gesamterlös minus Zielwert ist der **Netto-Erlös**). Das **Prize-Collecting-Steiner-Problem** ist **NP-schwer**; ohne Erlöse ist der leere Baum die Antwort, mit sehr hohen Erlösen der Steinerbaum über alle Kunden aus der [steiner-tree-demo](../steiner-tree-demo). Die Demo misst, **was die Auswahl bringt**, wie gut das **Primal-Dual-Verfahren von Goemans und Williamson** (GW) mit seiner **Untergrenze** gegen das **exakte Optimum** (Dreyfus-Wagner, nur für wenige Kunden) abschneidet, was **starkes Beschneiden** ändert, ob **"erst Steinerbaum, dann kürzen"** oder eine **Lokalsuche** besser ist, und ob die Menge der angeschlossenen Kunden mit dem Erlösniveau **nur wächst**. Der Kernsatz macht das Optimum überprüfbar: jede Lösung ist ein Baum auf einer Knotenmenge X mit Depot, also ist **OPT = min über X von MST(G[X]) + Erlöse der Kunden außerhalb von X**.

**Einordnung in die Reihe:** geplant sind elf Stücke, dies ist das neunte:

```
Kruskal (Wurzel)                                                                           [gebaut: kruskal-demo]
 ├─ Prim (Kontrast: wächst von einem Punkt)                                                [gebaut: prim-demo]
 ├─ Borůvka (Kontrast: alle Komponenten parallel)                                          [gebaut: boruvka-demo]
 ├─ Euklidischer MST (keine n²-Kantenliste, Delaunay)                                      [gebaut: euclidean-mst-demo]
 ├─ Gerichteter Spannbaum (Chu-Liu/Edmonds)                                                [gebaut: arborescence-demo]
 ├─ Bottleneck-/Grad-/Hop-beschränkter Spannbaum                                           [gebaut: constrained-mst-demo]
 │    └─ Kapazitierter MST                                                                 [gebaut: cmst-demo]
 ├─ Steiner-Baum                                                                           [gebaut: steiner-tree-demo]
 │    └─ Prize-Collecting Steiner-Baum                                                     [DIESES STÜCK]
 ├─ MST-Sensitivität & dynamischer MST                                                     [nicht gebaut]
 └─ Zufällige Spannbäume & Kirchhoff                                                       [nicht gebaut]
```

Ergebnis in Kürze: **Die Auswahl lohnt nur mitten im Übergang: bei Niveau 2.5 (Plan 8 x 8, 8 Kunden) bringt das Optimum gegenüber der besseren der Grundlinien "alle" und "nichts anschließen" in 90 % der Instanzen etwas (im Mittel 6.0 %), bei Niveau 1.5 in 18 % und bei Niveau 4 in 58 %. GW ist kein gutes Näherungsverfahren für diese Aufgabe: mit GW-Beschneiden nie optimal (mittlere Lücke 12.8 %), mit starkem Beschneiden in 30 % (2.83 %); "erst Steinerbaum, dann kürzen" liegt nur 0.83 % über dem Optimum und schlägt GW in 58 % der Instanzen (GW es in 8 %). GW liefert dafür eine Untergrenze (im Mittel 68 % des Optimums) und eine Garantie, die nie verletzt wurde. Die Approximationsgüte betrifft den Zielwert, nicht den Netto-Erlös: 2.83 % Zielwert-Lücke von GW (stark) sind im Mittel 24.6 % Netto-Erlös-Verlust.**

| Frage | Ergebnis (Plan 8 x 8, 8 Kunden, 10 % gesperrte Straßen, gleichverteilt, gleiche Erlöse, Niveau 2.5, sofern nicht anders angegeben; **50 Instanzen**, Seeds 200000–200049, bzw. **Median** über 5 feste Instanzen, Seeds 100000–100004; vollständig deterministisch) |
|---|---|
| **Ist der Kernsatz und Dreyfus-Wagner richtig?** | ✅ ja, direkt geprüft: Dreyfus-Wagner gleich dem Minimum über **alle** Knotenmengen (Brute-Force auf 160 Zufallsgraphen mit Gleichständen, dazu 120 Pläne 3 x 3); der Kernsatz zusätzlich gegen eine davon unabhängige Aufzählung **aller Bäume** durch das Depot |
| **Wo lohnt sich die Auswahl?** | Wert der Auswahl (1 − Optimum / bessere Grundlinie) bei Niveau 1.5/2.5/4: in **18/90/58 %** der Instanzen etwas, im Mittel **1.1/6.0/1.8 %**; angeschlossene Kunden im Mittel 0.4/5.5/7.8 von 8. Median über 5 feste Instanzen bei Niveau 1.5/2/2.5/3/4/6: **0/62.5/87.5/100/100/100 %** der Kunden angeschlossen (leerer Baum → alle) |
| **Alle oder nichts anschließen** | Bei Niveau 2.5 liegt "alle anschließen" im Mittel **9.1 %** über dem Optimum (Zielwert), "nichts anschließen" **16.5 %**; bei Niveau 1.5 "alle" +59.6 %, bei Niveau 4 "nichts" +74.6 % |
| **Gruppierte Kunden** | ⚠️ der Übergang liegt tiefer: 9 Kunden in Gruppen, Niveau 1.5: Auswahl lohnt in **78 %** (im Mittel 5.1 %), 6.1 von 9 Kunden angeschlossen; Niveau 2.5: nur **10 %** (0.17 %), 8.7 von 9 angeschlossen |
| **Sperrungen, gemischte Erlöse** | Bei 0/10/30 % gesperrten Straßen werden im Mittel **6.3/5.5/3.0** von 8 Kunden angeschlossen (Auswahl lohnt in 78/90/60 %); gemischte Erlöse (0.5 bis 1.5 x) verhalten sich wie gleiche: 84 %, 6.8 %, 4.9 Kunden |
| **Wie gut ist GW?** | Bei Niveau 2.5: **GW-Beschneiden** nie optimal (mittlere Lücke **12.79 %**, größte 28.36 %), **starkes Beschneiden** in 30 % optimal (**2.83 %**, größte 12.39 %); der rohe feste Baum durch das Depot ist mit +164.5 % unbrauchbar. Garantie 2 − 1/(n − 1) in **keiner** gemessenen Instanz verletzt |
| **Wie hängt GW vom Niveau ab?** | Niveau 1.5: GW-Beschneiden **+22.0 %** (34 % optimal), starkes Beschneiden **0.00 %** (100 % optimal): das grobe Beschneiden lässt zu viele Kunden stehen; Niveau 4: beide bei **6.8/6.1 %** – dort ist GW als Steinerbaum-Verfahren schwach (2-Approximation). Starkes Beschneiden ist besser als GW-Beschneiden in **66/88/24 %** der Instanzen (Niveau 1.5/2.5/4) |
| **Erst Steinerbaum, dann kürzen** | Mittlere Lücke **0.83 %** (größte 4.18 %, in 56 % optimal) bei Niveau 2.5; Niveau 4: 0.87 %. Schlägt GW (stark) in **58 %** der Instanzen, GW (stark) schlägt es in **8 %** (Niveau 4: 86 % gegen 0 %) |
| **Lokalsuche** | Mittlere Lücke **0.69 %** (größte 4.18 %, 60 % optimal); sie verbessert den Start (bestes von GW stark / erst Steiner) in **0 %** der Instanzen – ihr Vorsprung kommt nur daraus, dass sie vom besseren der beiden Starts ausgeht (mit gemischten Erlösen 4 %) |
| **Wie gut ist die Untergrenze?** | Untergrenze / Optimum im Mittel **0.93/0.68/0.64** bei Niveau 1.5/2.5/4 (kleinstes 0.58 bei 2.5): nah am Optimum, solange fast nichts angeschlossen wird, weit darunter, wenn der Steinerbaum die Kosten bestimmt |
| **Zielwert gegen Netto-Erlös** | ⚠️ kleine Zielwert-Lücken sind große Netto-Verluste: GW (stark) **2.83 % → 24.6 %**, erst Steiner **0.83 % → 9.7 %**, Lokalsuche 0.69 % → 9.0 %, GW-Beschneiden **12.8 % → 163.8 %** (Netto-Erlös des Optimums nur 35 im Preset-Standardfall bei Gesamterlös 200) |
| **Wächst die Anschlussmenge nur?** | Der gesammelte Erlös des Optimums fällt mit dem Niveau **nie** (0 Ausnahmen). Die Menge selbst ist auf dem Plan 8 x 8 in **0 von 50** Instanzen genestet gewachsen (gleiche und gemischte Erlöse), auf Plan 10 x 10 mit 10 Kunden verletzt sie sich in **2 von 50** (Niveau 3.25 → 3.5); das Preset zeigt Seed 12 (gemischt): Kunde 58 fällt heraus, obwohl sein Erlös steigt. Ein Gegenbeispiel auf 6 Knoten ist per Brute-Force über alle Bäume bestätigt |
| **Aufwand des exakten Wegs** | Zustände (Teilmenge, Kreuzung) = (2^t − 1) · n, auf dem 8 x 8-Plan bei t = 4/8/12: 960/16 320/262 080; gemessen ca. 2 s bei t = 12, 5 s bei t = 13 – deshalb nur bis t = 12 |

## Was die Demo zeigt

1. **Der Baum in Aktion** (Schritt-Slider): **Der Plan** (Kreuzungen, Straßen, gesperrte Straßen rot gepunktet, Kunden als Kreise mit Größe = Erlös, Depot ⭐) → **Goemans-Williamson wächst** (Slider über die Ereignisse: **Moats** als Kreise um jeden Knoten – orange wachsende Komponenten, grau die, die nicht mehr wachsen –, bisher feste Straßen grün, die Straße, die jetzt fest wird, orange, eine Komponente, deren Erlös aufgebraucht ist, rot umrandet; Text mit Zeit, Zahl wachsender Komponenten und Untergrenze bisher) → **Wer wird angeschlossen** (Umschalter Bester Fund / Exakt / GW mit GW-Beschneiden / GW mit starkem Beschneiden / GW roh / erst Steiner, dann kürzen / Lokalsuche / alle / nichts; nicht angeschlossene Kunden grau und hohl, Rauten = Steinerknoten; darunter Netto-Erlös je Verfahren als Balken und eine Tabelle mit Kosten, Netto-Erlös, Zielwert, Lücke und Netto-Verlust).
2. **Kennzahlen:** Netto-Erlös des besten Baums, angeschlossene Kunden mit **Wert der Auswahl**, Lücke von GW (stark) und "erst Steiner" gegen die Referenz (Optimum bei höchstens 12 Kunden, sonst bester Fund), Untergrenze und Garantie.
3. **Vom leeren Baum zu allen Kunden:** Netto-Erlös in Prozent des Gesamterlöses je Verfahren über alle Erlösniveaus, Balken = im Optimum angeschlossene Kunden; darunter, ab welchem Niveau überhaupt ein Kunde und ab welchem alle angeschlossen werden.
4. **🔬 Auf Abruf:** Qualitätsexperiment über 50 Instanzen (Anteil optimal je Verfahren, Lücken, Untergrenze, Netto-Verlust, Garantieprüfung), Nestungs-Experiment (alle Niveaus je Instanz), Aufwandskurve des exakten Wegs, Sweeps über Niveau / Kundenzahl / Plangröße / Sperranteil / Lage / Erlösmodus (Lücke, Netto-Verlust, Wert der Auswahl, Untergrenze).

Presets (9): Standardfall, Lehrbuchbeispiel, niedriges und hohes Erlösniveau, gruppierte Kunden, GW weit über dem Optimum, "erst Steiner" verliert gegen GW, Lokalsuche bleibt hängen, Nestung verletzt.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Standardfall** | Plan 8 x 8, 8 Kunden, Niveau 2.5, Seed 10 | Optimum: 5 von 8 Kunden, Kosten 90.15, Netto-Erlös 34.85, Zielwert 165.15; alle anschließen 177.51 (+7.49 %), nichts 200 (+21.10 %), Wert der Auswahl 6.96 %; erst Steiner und Lokalsuche optimal, GW stark +1.07 %, GW-Beschneiden +8.56 % (alle 8 Kunden) |
| **Lehrbuchbeispiel** | Gabel mit Ausläufer, 7 Kreuzungen, A und B zahlen 3, C zahlt 0.6 | Stamm 3 + zwei Äste 1: einzeln 4 > 3, gemeinsam 5 < 6; Optimum A und B, Kosten 5, Zielwert 5.6, Netto-Erlös 1; alle 6, nichts 6.6; Untergrenze 5.6 = Optimum |
| **Niedriges Erlösniveau** | Niveau 1.5, Seed 37 | Optimum 3 von 8 (105.60, Wert der Auswahl 12.00 %); alle +38.19 %; GW-Beschneiden 7 Kunden, +34.35 %; stark 2 Kunden, +4.04 % |
| **Hohes Erlösniveau** | Niveau 6, Seed 0 | alle 8 lohnen, Optimum = Steinerbaum 196.73; nichts +143.99 %; GW +12.88 %; Untergrenze 64 % |
| **Gruppierte Kunden** | 9 Kunden in Gruppen, Niveau 1.5, Seed 47 | Optimum 5 von 9 (108.62), alle +38.17 %, nichts +24.29 %, Wert der Auswahl 19.54 %; GW-Beschneiden +7.16 % |
| **GW weit über dem Optimum** | Niveau 2.5, Seed 1 | GW stark 175.03 (+14.11 %), GW-Beschneiden 182.13 (+18.74 %) gegen 153.39; Netto-Erlös 46.61 → 24.97 (−46.4 %); erst Steiner +0.12 %; Untergrenze 70 % |
| **Erst Steiner verliert gegen GW** | Niveau 2.5, Seed 38 | erst Steiner 134.97 (+5.77 %) gegen 127.60; GW und Lokalsuche optimal; Netto-Erlös 72.40 → 65.03 (−10.2 %) |
| **Lokalsuche bleibt hängen** | Niveau 2.5, Seed 60 | Lokalsuche 174.38 (+5.68 %) gegen 165.01, ebenso erst Steiner; GW stark +9.13 % |
| **Nestung verletzt** | gemischte Erlöse, Seed 12, Niveau 3.25 (ab 3) | Niveau 3: 6 Kunden angeschlossen (darunter 58), Niveau 3.25: 7, aber ohne 58; 12 und 22 kommen dazu |

## Modell und Verfahren

- **Instanz** (`pcst_scenario.py`): der Stadtplan der Steiner-Baum-Demo (gestörtes Gitter, gesperrte Straßen unter Wahrung des Zusammenhangs, Kunden gleichverteilt oder in Gruppen; Depot = kleinster Knoten der Terminals – gleicher Plan wie dort bei t + 1 Terminals, per Test gegen deren Zahlen geprüft), dazu **Erlöse**: Niveau x Straßenlänge (Niveau 3 = drei Nachbarstraßen), gleich oder je Kunde 0.5- bis 1.5-fach (eigener Zufallsstrom).
- **Zielwert und Referenz** (`pcst_algorithm.py`): Zielwert = Kosten + Erlöse der nicht angeschlossenen Kunden; Referenz = exaktes Optimum bei höchstens 12 Kunden, sonst der beste Fund. Die **Netto-Lücke** ist der Verlust an Netto-Erlös gegen die Referenz in Prozent des Referenz-Netto-Erlöses.
- **Goemans-Williamson** (`gw_primal_dual`): jede Komponente ohne Depot ist aktiv, solange ihr Restpotenzial (Erlös minus Summe der Moats in ihr) positiv ist; aktive Komponenten laden alle ihre Knoten mit der Zeit auf. Ereignisse: eine Straße wird fest, wenn die Ladungen ihrer Enden ihre Länge erreichen (Rate 2 bei zwei aktiven Seiten, 1 bei einer), oder eine Komponente stirbt (Restpotenzial 0). Beim Verschmelzen addieren sich die Restpotenziale; eine Komponente mit Depot wächst nie. Die **Untergrenze** ist die Summe aller Moat-Variablen. **GW-Beschneiden** (`gw_prune`) entfernt so viel wie möglich, ohne dass ein Knoten, der nie in einer toten Menge lag, vom Depot getrennt wird oder eine tote Menge halb angeschlossen bleibt (Beschreibung nach dem Anhang von arXiv 1710.07040; zusätzlich werden übrig gebliebene Nicht-Kunden-Blätter abgeschnitten); **starkes Beschneiden** (`strong_prune`) ist das optimale Zurechtschneiden des festen Baums per Baum-DP (ein Teilbaum bleibt nur, wenn er mehr einbringt, als seine Anschlusskante kostet; Gleichstand: weg).
- **Erst Steinerbaum, dann kürzen** (`steiner_then_prune`): bester Steinerbaum über Depot und alle Kunden (KMB, Takahashi-Matsuyama mit jedem Terminal als Wurzel, Lokalsuche; Bausteine aus der Steiner-Baum-Demo), danach `strong_prune`.
- **Lokalsuche** (`local_search`): Knotenmenge X mit Depot, Wert = Zielwert des beschnittenen MST des induzierten Teilgraphen; beste Verbesserung durch Einfügen oder Entfernen eines Knotens, streng fallend; Start = besseres von GW (stark) und erst Steiner.
- **Exakt** (`pcst_exact`, `ExactTables`): Dreyfus-Wagner über alle t Kunden mit dem Depot als Wurzel; OPT = min über Kundenteilmengen S von dp[S][Depot] + Erlöse der Kunden außerhalb von S. Die Tabelle hängt nicht von den Erlösen ab, darum kostet die Kurve über alle Niveaus eine einzige Rechnung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Hypothese "GW ist nahe am Optimum" – widerlegt.** Mit GW-Beschneiden ist GW bei Niveau 2.5 in keiner der 50 Instanzen optimal (mittlere Lücke 12.8 %), mit starkem Beschneiden in 30 % (2.83 %); "erst Steinerbaum, dann kürzen" (0.83 %) und die Lokalsuche (0.69 %) sind besser. GW bleibt wegen der **Untergrenze** und der **bewiesenen Garantie** lehrreich, nicht wegen der Güte in der Praxis.
- **Vorab-Hypothese "Beschneiden bringt wenig" – widerlegt.** Der rohe feste Baum ist mit +164.5 % Zielwert unbrauchbar; GW-Beschneiden kostet bei Niveau 1.5 noch 22.0 % (starkes Beschneiden 0.00 %). Ein Nebenbefund: bei Niveau 4 und höher sind beide Beschneidungen gleich gut (6.1 bis 6.8 %), der Rest der Lücke ist die Schwäche von GW als Steinerbaum-Verfahren.
- **Vorab-Hypothese "die Anschlussmenge ist genestet" – auf dem Plan bestätigt, im Allgemeinen widerlegt.** Der gesammelte Erlös des Optimums fällt nie (Austauschargument, in allen Messungen 0 Ausnahmen), die Menge kann aber wechseln, weil die Kosten des Steinerbaums keine einfache Summe über die Kunden sind. Häufigkeit auf den Plänen: 0 von 50 (8 x 8), 2 von 50 (10 x 10, 10 Kunden); ein Gegenbeispiel auf 6 Knoten ist durch Brute-Force bestätigt.
- **Beschneiden nach dem Anhang einer Übersichtsarbeit.** Die Regel für GW-Beschneiden folgt der Kurzbeschreibung im Anhang von arXiv 1710.07040 (nicht dem Original von 1995); die Tests (Untergrenze ≤ Optimum, Garantie gegen das exakte Optimum auf 320 Zufallsgraphen und 30 Plänen) bestätigen sie, ersetzen aber nicht den Vergleich mit dem Originaltext. Laut dieser Arbeit enthält der Güte-Beweis von Johnson u. a. für die **ungewurzelte** Variante einen Fehler (Feofiloff u. a.); hier wird nur die **gewurzelte** Variante gebaut, für die das starke Beschneiden nur ein optimales Zuschneiden des festen Baums ist.
- **Zielwert ≠ Netto-Erlös.** Die Approximationsgarantie von GW gilt für den Zielwert (Kosten + Erlöse der Nichtangeschlossenen); der Netto-Erlös kann relativ viel stärker leiden (2.83 % → 24.6 %). Verfahren mit Garantie für den Netto-Erlös sind nicht gebaut.
- **Exakt nur bis 12 Kunden.** Darüber ist die Referenz der beste Fund (die Lücken der Verfahren sind dann untereinander, nicht gegen das Optimum gemessen). Der Stand der Technik: die **1.79-Approximation** (Ahmadi u. a., STOC 2024, nur genannt), Branch-and-Cut und Reduktionen (nicht gebaut).
- **Einfache GW-Simulation.** Jedes Ereignis durchsucht alle Kanten (O(n · m)); die beschleunigte Implementierung von Johnson u. a. ist nicht gebaut. Zeit ist hier nicht der Gegenstand.
- **Synthetisches Modell.** Gestörtes Gitter, Länge als einzige Kosten, Erlöse additiv und unabhängig, ein Depot; keine Kapazität, keine Mindestmengen, keine Kosten für Steinerpunkte. Die Ergebnisse gelten für diese Pläne (bis 14 x 14, bis 30 Kunden), nicht für reale Netze; die Ursachen der Unterschiede (Steinerbaum-Schwäche von GW, grobes Beschneiden) sind gemessen, nicht einzeln isoliert.

## Verifikation

- `tests/test_algorithm.py`: Kernsatz und Exaktheit gegen Brute-Force (alle Bäume, alle Knotenmengen, 160 Zufallsgraphen + 120 Pläne, Sonderfälle: Erlöse 0 → leerer Baum, riesige Erlöse → Steinerbaum, Kunde genau an der Schwelle), Gültigkeit aller Verfahren mit unabhängiger Neuberechnung, `strong_prune` gleich dem besten Wurzel-Teilbaum (80 Bäume), GW: **Untergrenze ≤ Optimum und Garantie 2 − 1/(n − 1) gegen exakt auf 320 Zufallsgraphen**, Buchführung der Ereignisse (Zeiten, Komponentenzahl, Integral der aktiven Komponenten), Lehrbuchbeispiel von Hand (Untergrenze 5.6, Zielwert 5.6), Schrankenkette, Lokalsuche als lokales Optimum, Nestungs-Gegenbeispiel per Brute-Force über alle Bäume.
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (Bänder + jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt und jedes Ereignis, alle Bäume, Randwerte, Würfel, Permalink-Grenzen, Instanzwechsel, Experimente und Sweeps auf Abruf, Footer).
- Für die exakten Rechnungen genügt **pytest** (Brute-Force und eigener Dreyfus-Wagner tragen die Prüfung; kein scipy/networkx).

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Goemans, M. X., & Williamson, D. P. (1995). *A general approximation technique for constrained forest problems.* SIAM Journal on Computing 24(2), 296–317.
- Johnson, D. S., Minkoff, M., & Phillips, S. (2000). *The prize collecting Steiner tree problem: theory and practice.* Proc. 11th ACM-SIAM Symposium on Discrete Algorithms, 760–769.
- Ahmadi, A., Gholami, I., Hajiaghayi, M., Jabbarzade, P., & Mahdavi, M. (2024). *Prize-collecting Steiner tree: a 1.79 approximation.* STOC 2024, arXiv 2405.03792 (nur genannt, nicht gebaut).
- Bausteine des Steinerbaums (wie in der Steiner-Baum-Demo): Takahashi & Matsuyama (1980), Kou, Markowsky & Berman (1981), Dreyfus & Wagner (1971).
- Zur Beschreibung von GW-Wachstum und -Beschneiden: arXiv 1710.07040 (Anhang A, Übersicht über GW, Johnson u. a. und Feofiloff u. a.).

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
