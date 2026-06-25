import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation import simulate_gbm
from src.risk import var_historique, sharpe_ratio

def test_gbm_shape():
    paths = simulate_gbm(S0=100, mu=0.08, sigma=0.2, T=252, n_simulations=500)
    assert paths.shape == (252, 500)

def test_gbm_positive_prices():
    paths = simulate_gbm(S0=100, mu=0.08, sigma=0.2, T=252, n_simulations=200)
    assert np.all(paths > 0)

def test_var_negative():
    returns = np.random.normal(-0.01, 0.02, 1000)
    var = var_historique(returns)
    assert var < 0

def test_sharpe_positive():
    returns = np.random.normal(0.001, 0.01, 252)
    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)