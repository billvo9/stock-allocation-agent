import numpy as np

from stock_agent.evaluation.baseline import drift_weights

weights = np.array([0.50, 0.30, 0.20])
returns = np.array([0.10, 0.00, -0.10])

def test_drift_weights():
    result = drift_weights(weights, returns)
    assert np.isclose(result.sum(), 1.0)