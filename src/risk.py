import numpy as np

def var_historique(returns: np.ndarray, alpha: float = 0.05) -> float:
    """VaR historique au niveau de confiance (1-alpha)."""
    return np.percentile(returns, alpha * 100)

def var_parametrique(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """VaR paramétrique (hypothèse normale)."""
    from scipy.stats import norm
    return mu - sigma * norm.ppf(1 - alpha)

def var_monte_carlo(simulated_paths: np.ndarray, alpha: float = 0.05) -> float:
    """VaR Monte-Carlo sur les rendements finaux simulés."""
    final_values = simulated_paths[-1]
    initial_value = simulated_paths[0].mean()
    returns = (final_values - initial_value) / initial_value
    return np.percentile(returns, alpha * 100)

def cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR (Expected Shortfall) : moyenne des pertes au-delà de la VaR."""
    var = var_historique(returns, alpha)
    return returns[returns <= var].mean()

def sharpe_ratio(returns: np.ndarray, rf: float = 0.02) -> float:
    """Sharpe Ratio annualisé."""
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return (mu - rf) / sigma

def max_drawdown(prices: np.ndarray) -> float:
    """Maximum Drawdown sur une série de prix."""
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return drawdown.min()