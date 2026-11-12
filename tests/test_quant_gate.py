import numpy as np

from ml.quant_gate import gate


def _data():
    y = np.array([1] * 30 + [0] * 30 + [2] * 30)
    groups = np.array((["I-II"] * 10 + ["III-IV"] * 10 + ["V-VI"] * 10) * 3)
    return y, groups


def test_identical_predictions_pass():
    y, groups = _data()
    assert gate(y.copy(), y.copy(), y, groups, "int8", 1.0).passed


def test_losing_malignant_cases_in_one_group_fails():
    y, groups = _data()
    q = y.copy()
    q[20:23] = 2  # 3 malignos de V-VI mal clasificados -> recall cae 0.3 en V-VI
    g = gate(y.copy(), q, y, groups, "int8", 1.0)
    assert not g.passed and g.malignant_recall_drop_by_group["V-VI"] > 0.02
