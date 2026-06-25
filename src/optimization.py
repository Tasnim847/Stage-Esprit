import numpy as np
from scipy.optimize import minimize

def portfolio_performance(weights, mu, cov_matrix):
    ret = np.dot(weights, mu)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

def max_sharpe(mu, cov_matrix, rf=0.02):
    """Trouve le portefeuille avec le Sharpe Ratio maximal."""
    n = len(mu)
    def neg_sharpe(w):
        r, v = portfolio_performance(w, mu, cov_matrix)
        return -(r - rf) / v

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints)
    return result.x

def min_variance(mu, cov_matrix):
    """Trouve le portefeuille à variance minimale."""
    n = len(mu)
    def portfolio_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    result = minimize(portfolio_vol, w0, bounds=bounds, constraints=constraints)
    return result.x

def efficient_frontier(mu, cov_matrix, n_points=100):
    """Génère les points de la frontière efficiente."""
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    frontier_vols = []

    for target in target_returns:
        n = len(mu)
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mu) - t}
        ]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n
        res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                       w0, bounds=bounds, constraints=constraints)
        frontier_vols.append(res.fun if res.success else np.nan)

    return target_returns, np.array(frontier_vols)