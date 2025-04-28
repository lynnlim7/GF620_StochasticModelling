######## Imports
import numpy as np
import scipy.stats as stats

######## Bachelier Functions
def BachelierCall(S: float, K: float, r: float, sigma: float, T: float) -> float:
    val_diff = S - K
    vol_sqrt_time = sigma * np.sqrt(T)
    z_score = val_diff / vol_sqrt_time
    discount_factor = np.exp(-r * T)

    return (discount_factor *
            (val_diff * stats.norm.cdf(z_score) + vol_sqrt_time * stats.norm.pdf(z_score)))

def BachelierPut(S: float, K: float, r: float, sigma: float, T: float) -> float:
    val_diff = K - S
    vol_sqrt_time = sigma * np.sqrt(T)
    z_score = val_diff / vol_sqrt_time
    discount_factor = np.exp(-r * T)

    return (discount_factor *
            (val_diff * stats.norm.cdf(z_score) + vol_sqrt_time * stats.norm.pdf(z_score)))

def BachelierCashOrNthCall(S: float, K: float, r: float, sigma: float, T: float, Q: float) -> float:
    val_diff = S - K
    vol_sqrt_time = sigma * np.sqrt(T)
    z_score = val_diff / vol_sqrt_time
    discount_factor = np.exp(-r * T)
    return Q * discount_factor * stats.norm.cdf(z_score)

def BachelierCashOrNthPut(S: float, K: float, r: float, sigma: float, T: float, Q: float) -> float:
    val_diff = K - S
    vol_sqrt_time = sigma * np.sqrt(T)
    z_score = val_diff / vol_sqrt_time
    discount_factor = np.exp(-r * T)
    return Q * discount_factor * stats.norm.cdf(z_score)

def BachelierAssetOrNthCall(S: float, K: float, r: float, sigma: float, T: float) -> float:
    val_diff = S - K
    vol_sqrt_time = sigma * np.sqrt(T)
    z_score = val_diff / vol_sqrt_time
    discount_factor = np.exp(-r * T)
    return (discount_factor *
            (S * stats.norm.cdf(z_score) + vol_sqrt_time * stats.norm.pdf(z_score)))

def BachelierAssetOrNthPut(S: float, K: float, r: float, sigma: float, T: float) -> float:
    val_diff = K - S
    vol_sqrt_time = sigma * np.sqrt(T)
    z_score = val_diff / vol_sqrt_time
    discount_factor = np.exp(-r * T)
    return (discount_factor *
            (S * stats.norm.cdf(z_score) + vol_sqrt_time * stats.norm.pdf(z_score)))

######## Black Scholes Functions
def calc_black_scholes_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    sigma_sqrt_time = sigma * np.sqrt(T)
    return ( np.log(S / K) + (r + np.power(sigma, 2)/2) * T ) / sigma_sqrt_time

def calc_black_scholes_d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    sigma_sqrt_time = sigma * np.sqrt(T)
    return calc_black_scholes_d1(S, K, r, sigma, T)  - sigma_sqrt_time

def BlackScholesCall(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_scholes_d1(S, K, r, sigma, T)
    d2 = calc_black_scholes_d2(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return S * stats.norm.cdf(d1)  - K * discount_factor * stats.norm.cdf(d2)
    
def BlackScholesPut(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_scholes_d1(S, K, r, sigma, T)
    d2 = calc_black_scholes_d2(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return K * discount_factor * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

def BlackScholesCashOrNthCall(S: float, K: float, r: float, sigma: float, T: float, Q: float) -> float:
    d2 = calc_black_scholes_d2(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return Q * discount_factor * stats.norm.cdf(d2)

def BlackScholesCashOrNthPut(S: float, K: float, r: float, sigma: float, T: float, Q: float) -> float:
    d2 = calc_black_scholes_d2(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return Q * discount_factor * stats.norm.cdf(-d2)

def BlackScholesAssetOrNthCall(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_scholes_d1(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return S * discount_factor * stats.norm.cdf(d1) 

def BlackScholesAssetOrNthPut(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_scholes_d1(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return S * discount_factor * stats.norm.cdf(-d1)

######## Black  Functions
def calc_black_model_d1(F: float, K: float, sigma: float, T: float) -> float:
    sigma_sqrt_time = sigma * np.sqrt(T)
    num1 = np.log(F / K)
    num2 = 0.5 * np.power(sigma, 2) * T
    return ( num1 + num2 ) / sigma_sqrt_time

def calc_black_model_d2(F: float, K: float, sigma: float, T: float) -> float:
    sigma_sqrt_time = sigma * np.sqrt(T)
    d1 = calc_black_model_d1(F, K, sigma, T)
    return d1 - sigma_sqrt_time

def BlackCall(F: float, K: float, r: float, sigma: float, T: float) -> float:
    discount_factor = np.exp(-r * T)
    d1 = calc_black_model_d1(F, K, sigma, T)
    d2 = calc_black_model_d2(F, K, sigma, T)
    return discount_factor * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))

def BlackPut(F: float, K: float, r: float, sigma: float, T: float) -> float:
    discount_factor = np.exp(-r * T)
    d1 = calc_black_model_d1(F, K, sigma, T)
    d2 = calc_black_model_d2(F, K, sigma, T)
    return discount_factor * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))

def BlackCashOrNthCall(F: float, K: float, r: float, sigma: float, T: float, Q: float) -> float:
    d2 = calc_black_model_d2(F, K, sigma, T)
    discount_factor = np.exp(-r * T)
    return Q * discount_factor * stats.norm.cdf(d2)

def BlackCashOrNthPut(F: float, K: float, r: float, sigma: float, T: float, Q: float) -> float:
    d2 = calc_black_model_d2(F, K, sigma, T)
    discount_factor = np.exp(-r * T)
    return Q * discount_factor * stats.norm.cdf(-d2)

def BlackAssetOrNthCall(F: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_model_d1(F, K, sigma, T)
    discount_factor = np.exp(-r * T)
    return F * discount_factor * stats.norm.cdf(d1)

def BlackAssetOrNthPut(F: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_model_d1(F, K, sigma, T)
    discount_factor = np.exp(-r * T)
    return F * discount_factor * stats.norm.cdf(-d1)

######## Displaced Diffusion Functions
def calc_dd_f(F: float, B: float) -> float:
  return F / B

def calc_dd_k(F: float, K: float, B: float) -> float:
  return K + np.divide(1 - B, B) * F

def calc_dd_sigma(sigma: float, B: float) -> float:
  return sigma * B

def DisplacedDiffusionCall(F: float, K: float, r: float, sigma: float, T: float, B: float) -> float:
  return BlackCall( F = calc_dd_f(F,B), K = calc_dd_k(F,K,B), r=r, sigma = calc_dd_sigma(sigma,B), T = T)

def DisplacedDiffusionPut(F: float, K: float, r: float, sigma: float, T: float, B: float) -> float:
  return BlackPut( F = calc_dd_f(F,B), K = calc_dd_k(F,K,B), r=r, sigma = calc_dd_sigma(sigma,B), T = T)

def DisplacedDiffusionCashOrNthCall(F: float, K: float, r: float, sigma: float, T: float, B: float, Q: float) -> float:
  return BlackCashOrNthCall( F = calc_dd_f(F,B), K = calc_dd_k(F,K,B), r=r, sigma = calc_dd_sigma(sigma,B), T = T, Q = Q)

def DisplacedDiffusionCashOrNthPut(F: float, K: float, r: float, sigma: float, T: float, B: float, Q: float) -> float:
  return BlackCashOrNthPut( F = calc_dd_f(F,B), K = calc_dd_k(F,K,B), r=r, sigma = calc_dd_sigma(sigma,B), T = T, Q = Q)

def DisplacedDiffusionAssetOrNthCall(F: float, K: float, r: float, sigma: float, T: float, B: float) -> float:
  return BlackAssetOrNthCall( F = calc_dd_f(F,B), K = calc_dd_k(F,K,B), r=r, sigma = calc_dd_sigma(sigma,B), T = T)

def DisplacedDiffusionAssetOrNthPut(F: float, K: float, r: float, sigma: float, T: float, B: float) -> float:
  return BlackAssetOrNthPut( F = calc_dd_f(F,B), K = calc_dd_k(F,K,B), r=r, sigma = calc_dd_sigma(sigma,B), T = T)


print("Loaded All Functions")
print("Bachelier Model Sample Calculations")
print(f"Strike > Intial val, Value of Call: {BachelierCall(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Call: {BachelierCall(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Put: {BachelierPut(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Put: {BachelierPut(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Call: {BachelierCashOrNthCall(S=100.0, r=0.05, sigma=0.2, K=105, T=1, Q=10):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Call: {BachelierCashOrNthCall(S=100.0, r=0.05, sigma=0.2, K=95, T=1, Q=10):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Put: {BachelierCashOrNthPut(S=100.0, r=0.05, sigma=0.2, K=105, T=1, Q=10):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Put: {BachelierCashOrNthPut(S=100.0, r=0.05, sigma=0.2, K=95, T=1, Q=10):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Call: {BachelierAssetOrNthCall(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Call: {BachelierAssetOrNthCall(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Put: {BachelierAssetOrNthPut(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Put: {BachelierAssetOrNthPut(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print("Black Scholes Model Sample Functions")
print(f"Strike > Intial val, Value of Call: {BlackScholesCall(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Call: {BlackScholesCall(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Put: {BlackScholesPut(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Put: {BlackScholesPut(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Call: {BlackScholesCashOrNthCall(S=100.0, r=0.05, sigma=0.2, K=105, T=1, Q=1):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Call: {BlackScholesCashOrNthCall(S=100.0, r=0.05, sigma=0.2, K=95, T=1, Q=1):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Put: {BlackScholesCashOrNthPut(S=100.0, r=0.05, sigma=0.2, K=105, T=1, Q=1):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Put: {BlackScholesCashOrNthPut(S=100.0, r=0.05, sigma=0.2, K=95, T=1, Q=1):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Call: {BlackScholesAssetOrNthCall(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Call: {BlackScholesAssetOrNthCall(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Put: {BlackScholesAssetOrNthPut(S=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Put: {BlackScholesAssetOrNthPut(S=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")


print("Black Model Sample Functions")
print(f"Strike > Intial val, Value of Call: {BlackCall(F=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Call: {BlackCall(F=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Put: {BlackPut(F=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Put: {BlackPut(F=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Call: {BlackCashOrNthCall(F=100.0, r=0.05, sigma=0.2, K=105, T=1, Q=1):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Call: {BlackCashOrNthCall(F=100.0, r=0.05, sigma=0.2, K=95, T=1, Q=1):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Put: {BlackCashOrNthPut(F=100.0, r=0.05, sigma=0.2, K=105, T=1, Q=1):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Put: {BlackCashOrNthPut(F=100.0, r=0.05, sigma=0.2, K=95, T=1, Q=1):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Call: {BlackAssetOrNthCall(F=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Call: {BlackAssetOrNthCall(F=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Put: {BlackAssetOrNthPut(F=100.0, r=0.05, sigma=0.2, K=105, T=1):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Put: {BlackAssetOrNthPut(F=100.0, r=0.05, sigma=0.2, K=95, T=1):.2f}")

print("Displaced Diffusion Model Sample Functions")
print(f"Strike > Intial val, Value of Call: {DisplacedDiffusionCall(F=100.0, r=0.05, sigma=0.2, K=105, T=1, B=0.7):.2f}")
print(f"Strike < Intial val, Value of Call: {DisplacedDiffusionCall(F=100.0, r=0.05, sigma=0.2, K=95, T=1, B=0.7):.2f}")

print(f"Strike > Intial val, Value of Put: {DisplacedDiffusionPut(F=100.0, r=0.05, sigma=0.2, K=105, T=1, B=0.7):.2f}")
print(f"Strike < Intial val, Value of Put: {DisplacedDiffusionPut(F=100.0, r=0.05, sigma=0.2, K=95, T=1, B=0.7):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Call: {DisplacedDiffusionCashOrNthCall(F=100.0, r=0.05, sigma=0.2, K=105, T=1, B=0.7, Q=1):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Call: {DisplacedDiffusionCashOrNthCall(F=100.0, r=0.05, sigma=0.2, K=95, T=1, B=0.7, Q=1):.2f}")

print(f"Strike > Intial val, Value of Digital Cash-Or-Nth Put: {DisplacedDiffusionCashOrNthPut(F=100.0, r=0.05, sigma=0.2, K=105, T=1, B=0.7, Q=1):.2f}")
print(f"Strike < Intial val, Value of Digital Cash-Or-Nth Put: {DisplacedDiffusionCashOrNthPut(F=100.0, r=0.05, sigma=0.2, K=95, T=1, B=0.7, Q=1):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Call: {DisplacedDiffusionAssetOrNthCall(F=100.0, r=0.05, sigma=0.2, K=105, T=1, B=0.7):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Call: {DisplacedDiffusionAssetOrNthCall(F=100.0, r=0.05, sigma=0.2, K=95, T=1, B=0.7):.2f}")

print(f"Strike > Intial val, Value of Digital Asset-Or-Nth Put: {DisplacedDiffusionAssetOrNthPut(F=100.0, r=0.05, sigma=0.2, K=105, T=1, B=0.7):.2f}")
print(f"Strike < Intial val, Value of Digital Asset-Or-Nth Put: {DisplacedDiffusionAssetOrNthPut(F=100.0, r=0.05, sigma=0.2, K=95, T=1, B=0.7):.2f}")

