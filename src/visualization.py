import plotly.graph_objects as go
import numpy as np

def plot_simulations(paths: np.ndarray, ticker: str = "Actif", n_display: int = 100):
    """Affiche les trajectoires simulées Monte-Carlo."""
    fig = go.Figure()
    for i in range(min(n_display, paths.shape[1])):
        fig.add_trace(go.Scatter(y=paths[:, i], mode='lines',
                                  line=dict(width=0.5),
                                  opacity=0.4, showlegend=False))
    fig.add_trace(go.Scatter(y=paths.mean(axis=1), mode='lines',
                              line=dict(color='red', width=2),
                              name='Moyenne'))
    fig.update_layout(title=f"Simulation Monte-Carlo — {ticker}",
                      xaxis_title="Jours", yaxis_title="Prix")
    fig.show()

def plot_frontier(target_returns, frontier_vols, opt_weights_label="Sharpe Max"):
    """Affiche la frontière efficiente."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frontier_vols, y=target_returns,
                              mode='lines', name='Frontière Efficiente',
                              line=dict(color='blue', width=2)))
    fig.update_layout(title="Frontière Efficiente de Markowitz",
                      xaxis_title="Volatilité (risque)",
                      yaxis_title="Rendement espéré")
    fig.show()

def plot_var_distribution(returns: np.ndarray, var_value: float, alpha: float = 0.05):
    """Affiche la distribution des rendements avec la VaR."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Rendements',
                                marker_color='steelblue', opacity=0.75))
    fig.add_vline(x=var_value, line_color='red', line_dash='dash',
                  annotation_text=f"VaR {int((1-alpha)*100)}%",
                  annotation_position="top right")
    fig.update_layout(title="Distribution des Rendements et VaR",
                      xaxis_title="Rendement", yaxis_title="Fréquence")
    fig.show()