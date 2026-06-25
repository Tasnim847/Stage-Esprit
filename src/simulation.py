import numpy as np
import pandas as pd

def simulate_gbm(S0: float, mu: float, sigma: float,
                 T: int = 252, n_simulations: int = 1000) -> np.ndarray:
    """
    Simulation Monte-Carlo par Mouvement Brownien Géométrique.
    S0 : prix initial
    mu : rendement annualisé
    sigma : volatilité annualisée
    T : nombre de jours
    n_simulations : nombre de trajectoires
    Retourne un tableau (T, n_simulations)
    """
    dt = 1 / 252
    prices = np.zeros((T, n_simulations))
    prices[0] = S0

    for t in range(1, T):
        Z = np.random.standard_normal(n_simulations)
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt
                                          + sigma * np.sqrt(dt) * Z)
    return prices

def simulate_portfolio(weights: np.ndarray, prices_dict: dict,
                       mu_vec: np.ndarray, sigma_vec: np.ndarray,
                       T: int = 252, n_simulations: int = 1000) -> np.ndarray:
    """Simule la valeur d'un portefeuille pondéré."""
    n_assets = len(weights)
    portfolio_values = np.zeros((T, n_simulations))

    for i, (ticker, S0) in enumerate(prices_dict.items()):
        paths = simulate_gbm(S0, mu_vec[i], sigma_vec[i], T, n_simulations)
        portfolio_values += weights[i] * paths

    return portfolio_values