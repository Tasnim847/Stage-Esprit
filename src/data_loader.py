import yfinance as yf
import numpy as np
import pandas as pd

def get_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Télécharge les prix de clôture ajustés."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data.dropna()

def get_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcule les rendements logarithmiques journaliers."""
    return np.log(prices / prices.shift(1)).dropna()

def get_stats(returns: pd.DataFrame):
    """Retourne mu (rendement annualisé) et sigma (volatilité annualisée)."""
    mu = returns.mean() * 252
    sigma = returns.std() * np.sqrt(252)
    return mu, sigma