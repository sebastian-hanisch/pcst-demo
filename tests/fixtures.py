"""Festgeschriebene Fixtures (per Skriptsuche gefunden, durch Brute-Force in test_algorithm bestätigt)."""

import numpy as np

import pcst_scenario as S

# Kleines Gegenbeispiel zur Nestung: Depot 0, Kunden 1-5; Erlöse = Einheitserlöse x Skala. Bei Skala 15.0 ist der optimale Baum {1, 4} (und 5 gehört nicht dazu),
# bei Skala 16.5 der Baum mit {2, 5} - kein optimaler Baum der höheren Skala enthält alle Kunden eines optimalen Baums der niedrigeren.
NEST_EDGES = ((0, 1, 2.558), (0, 2, 4.284), (0, 3, 5.115), (0, 4, 0.834), (3, 5, 5.093), (4, 5, 4.863))
NEST_UNIT_PRIZES = (0.0, 0.014, 0.632, 0.298, 0.023, 0.866)
NEST_SCALES = (15.0, 16.5)


def nesting_instance(scale):
    prizes = tuple(float(p) * scale for p in NEST_UNIT_PRIZES)
    return S.Instance(np.zeros((6, 2)), NEST_EDGES, 0, (1, 2, 3, 4, 5), prizes, 0, "random")
