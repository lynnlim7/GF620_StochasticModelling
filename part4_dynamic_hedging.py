"""
Authors : Brandon & Lynn
Description : Dynamic Hedging using Black-Scholes Repicating Strategy
"""

import pandas as pd
import numpy as np
from math import log, sqrt, exp
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from scipy.stats import norm
import statistics

# initial stock price
S0 = 100

# strike price
K = 100

# volatility
sigma = 0.2

# number of trades
n_trade1 = 21  #rebalance hedge daily over 21 trading days
n_trade2 = 84  #rebalance hedge 4 times a day over 21 trading days

# option's maturity
T = 1/12

# risk-free interest rate
r = 0.05

# number of simulations
path = 50000

def BlackScholesCall(S: float, K: float, r: float, sigma: float, T: float):
    """
    Black-Scholes Lognormal model for pricing European call options.

    Args:
    F(float): Forward price
    K(float): strike price
    r(float): risk-free interest rate
    sigma(float): volatility
    T(float): time to expiration (in years)

    Returns:
    Option Price
    """
    d1 = (log(S/K)+(r+sigma**2/2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return  S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def BrownianMotion_Simulation(paths: int, steps: int, T: float):
    """
    Simulation of browninan motion

    Args:
    paths(int): number of paths
    steps(int): number of steps
    T(float): time to expiration (in years

    Returns:
    t(array): time grid
    X(array): Brownian Motion
    """
    deltaT = T/steps
    t = np.linspace(0, T, steps+1)
    X = np.c_[np.zeros((paths, 1)),
                np.random.randn(paths, steps)]
    return t, np.cumsum(np.sqrt(deltaT) * X, axis=1)

def BlackScholes_Path(S: float, r: float, sigma: float, t: float, W_T:float):
    """
    Stock Price at maturity, St

    Args:
    S(float): spot price
    r(float): risk-free interest rate
    T(float): time step
    sigma(float): volatility
    W_T(float): brownian motion

    Returns:
    Stock Price
    """
    return S * np.exp((r-sigma**2/2)*t + sigma*W_T)

def phi(S: float, K: float, r: float, sigma: float, T: float, t:float):
    """
    Stock position value ϕSt

    Args:
    S(float): spot price
    K(float): strike price
    r(float): risk-free interest rate
    sigma(float): volatility
    T(float): maturity (in years)
    t(float): time step

    Returns:
    Stock position value

    """
    d1 = (np.log(S/K)+(r+sigma**2/2)*(T-t)) / (sigma*np.sqrt(T-t))
    return norm.cdf(d1)

def psi_Bt(S: float, K: float, r: float, sigma: float, T: float, t: float):
    """
    Bond position value ѰBt

    Args:
    S(float): spot price
    K(float): strike price
    r(float): risk-free interest rate
    sigma(float): volatility
    T(float): maturity (in years)
    t(float): time step

    Returns:
    Bond position value
    """
    d1 = (np.log(S/K)+(r+sigma**2/2)*(T-t)) / (sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return -K*np.exp(-r*(T-t))*norm.cdf(d2)

def HedgingStrategy(S0: float, K: float, r: float, sigma: float, T: float, t: float, W_T: np.ndarray):
    # stock price paths
    S_T = np.array(BlackScholes_Path(S0, r, sigma, t, W_T))
    n_paths, n_time_steps = S_T.shape

    rate_step = r * (T / n_time_steps)
    call_option = BlackScholesCall(S0, K, r, sigma, T)

    # initialize array to store pnl
    final_pnl = np.zeros(n_paths)

    # iterate through each path
    for i in range(n_paths):
        stock_pos = phi(S_T[i], K, r, sigma, T, t)
        bond_pos = psi_Bt(S_T[i], K, r, sigma, T, t)

        delta_S = np.diff(S_T[i])
        hedging_err = np.sum(stock_pos[:-1] * delta_S + bond_pos[:-1] * rate_step)

        # payoff for the final stock price in the path
        payoff = max(S_T[i, -1] - K, 0)
        final_pnl[i] = hedging_err + call_option - payoff

    return final_pnl

t1, W_T1 = BrownianMotion_Simulation(path, n_trade1, T)
t2, W_T2 = BrownianMotion_Simulation(path, n_trade2, T)

sim_1 = HedgingStrategy(S0, K, r, sigma, T, t1, W_T1)
sim_2 = HedgingStrategy(S0, K, r, sigma, T, t2, W_T2)

plt.hist(sim_1, bins=50, range=[-2,2], facecolor='orange', align='mid')
plt.title('Simulation 1 (N_Trades = 21)')
plt.xlabel('Final Profit & Loss')
plt.ylabel('Frequency')

n_trade1_mean = np.mean(sim_1)
n_trade1_stddev = np.std(sim_1)

plt.hist(sim_2, bins=50, range=[-1,1], facecolor='red', align='mid')
plt.title('Simulation 2 (N_Trades = 84)')
plt.xlabel('Final Profit & Loss')
plt.ylabel('Frequency')

n_trade2_mean = np.mean(sim_2)
n_trade2_stddev = np.std(sim_2)

