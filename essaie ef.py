import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Générer des rendements aléatoires pour les actifs
np.random.seed(42)
n_assets = 4
n_obs = 1000
return_vec = np.random.randn(n_assets, n_obs)

# Fonction pour calculer les rendements et les risques des portefeuilles
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

# Fonction pour minimiser le risque
def minimize_risk(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[0]

# Générer des portefeuilles aléatoires
def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

# Calculer les rendements moyens et la matrice de covariance
mean_returns = np.mean(return_vec, axis=1)
cov_matrix = np.cov(return_vec)

# Générer des portefeuilles
num_portfolios = 10000
risk_free_rate = 0.0178
results, weights = generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

# Tracer la frontière efficiente
plt.figure(figsize=(10, 7))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
plt.title('Frontière Efficiente')
plt.xlabel('Risque (Volatilité)')
plt.ylabel('Rendement')
plt.colorbar(label='Ratio de Sharpe')
plt.show()
