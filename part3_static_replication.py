"""
Part3 - Yating & Ann
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import warnings
warnings.filterwarnings('ignore')


spx_raw = pd.read_csv('SPX_options.csv')
zero_rates = pd.read_csv('zero_rates_20201201.csv')

def BlackScholesCall(S, K, r, sigma, T):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def BlackScholesPut(S, K, r, sigma, T):
    return BlackScholesCall(S, K, r, sigma, T) - S + K*np.exp(-r*T)

def BachelierCall(S, K, r, sigma, T):
    d1 = (S-K)
    d2 = sigma * np.sqrt(T)
    return np.exp(-r*T) * (d1 * norm.cdf(d1/d2) + d2 * norm.pdf(d1/d2))

def BachelierPut(S, K, r, sigma, T):
    d1 = (K-S)
    d2 = sigma * np.sqrt(T)
    return np.exp(-r*T) * (d1 * norm.cdf(d1/d2) + d2 * norm.pdf(d1/d2))

################## part II SPX data ##################

S = 3662.45
days_to_maturity = (pd.Timestamp('20210115') - pd.Timestamp('20201201')).days
T = days_to_maturity/365.0

zero_rate_curve = interp1d(zero_rates['days'], zero_rates['rate'])
r = zero_rate_curve(days_to_maturity).item()/100.0 # risk-free rate same as part II

################ ATM option volatility ################
spx = spx_raw[spx_raw['exdate'] == 20210115]
spx['strike_price'] = spx['strike_price']*0.001
spx['option_price'] = (spx['best_bid'] + spx['best_offer'])/2

F = S*np.exp(r*T) # forward price

atm_call_k = spx[(spx['cp_flag'] == 'C')
                  &(np.abs(spx['strike_price']- F)==(np.abs(spx['strike_price']- F)).min()) # look for the option with the strike price closest to the forward value F, and treat this as the ATM option.
                 ]['strike_price'].min()
atm_call_price = spx[(spx['cp_flag'] == 'C')
                    &(spx['strike_price']==atm_call_k)
                    ]['option_price'].values[0]
atm_call_vol_LN = brentq(lambda x: atm_call_price\
                                - BlackScholesCall(S, atm_call_k, r, x, T),
                                1e-12, 10.0)
atm_call_vol_N = brentq(lambda x: atm_call_price\
                                - BachelierCall(S, atm_call_k, r, x, T),
                                1.0, 1e5)

atm_put_k = spx[(spx['cp_flag'] == 'P')
                &(np.abs(spx['strike_price']- F)==(np.abs(spx['strike_price']- F)).min())
                ]['strike_price'].max()
atm_put_price = spx[(spx['cp_flag'] == 'P')
                    &(spx['strike_price']==atm_put_k)
                    ]['option_price'].values[0]
atm_put_vol_LN = brentq(lambda x: atm_put_price\
                                - BlackScholesPut(S, atm_put_k, r, x, T),
                                1e-12, 10.0)
atm_put_vol_N = brentq(lambda x: atm_put_price\
                                - BachelierPut(S, atm_put_k, r, x, T),
                                1.0, 1e5)

atm_vol_LN = (atm_call_vol_LN + atm_put_vol_LN)/2  # ATM vol for BlackScholes model
atm_vol_N = (atm_call_vol_N + atm_put_vol_N)/2  # ATM vol for Bachelier model


############## part II SABR model params ##############
alpha = 1.817
beta = 0.7
rho = -0.404
nu = 2.790

print(f"""
Assumptions:
S0 = {S}
T = {T}
r = {r}
sigma_LN = {atm_vol_LN}
sigma_N = {atm_vol_N}

SABR model parameters:
alpha = {alpha}
beta = {beta}
rho = {rho}
nu= {nu}
""")

## Contract 1 - under Black-Scholes model
def PriceUnderBlackScholes(S, r, sigma, T):
    ## parameter:
    # S: initial price of the contract
    # r: risk free rate
    # sigma: the volatility of the underlying assest
    # T: expire time

    S_T1 = S * np.exp(r * T - 0.5 * sigma * sigma * T )
    S_T2 = sigma ** 2 * T

    E_1 = S_T1 ** (1/3) * np.exp((1/18) * S_T2)
    E_2 = np.log(S_T1)

    V_0 = np.exp(-r * T) * (E_1 + 1.5 * E_2 + 10)
    return V_0

price_of_contract = PriceUnderBlackScholes(S, r, atm_vol_LN, T)
print('The price of the derivative contract using Black-Scholes is: %.9f' % price_of_contract)

## Contract 1 - under Bachelier model
def PriceUnderBachelier(S,r,sigma, T):

    E_1 = S ** (1/3) - (sigma **2 * T)/ (9 * S**(5/3))
    E_2 = np.log(S) - (sigma**2 * T) / (2 * S**2)

    V_0 = np.exp(-r * T) * (E_1 + 1.5 * E_2 + 10)
    return V_0

price_of_contract = PriceUnderBachelier(S, r,atm_vol_N, T)
print('The price of the derivative contract using Bachelier is: %.9f' % price_of_contract)

## Contract 1 - using static replication under SABR model
def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    if abs(F - K) < 1e-12:
        numer1 = (((1 - beta)**2)/24)*alpha*alpha/(F**(2 - 2*beta))
        numer2 = 0.25*rho*beta*nu*alpha/(F**(1 - beta))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        VolAtm = alpha*(1 + (numer1 + numer2 + numer3)*T)/(F**(1-beta))
        sabrsigma = VolAtm
    else:
        z = (nu/alpha)*((F*X)**(0.5*(1-beta)))*np.log(F/X)
        zhi = np.log((((1 - 2*rho*z + z*z)**0.5) + z - rho)/(1 - rho))
        numer1 = (((1 - beta)**2)/24)*((alpha*alpha)/((F*X)**(1 - beta)))
        numer2 = 0.25*rho*beta*nu*alpha/((F*X)**((1 - beta)/2))
        numer3 = ((2 - 3*rho*rho)/24)*nu*nu
        numer = alpha*(1 + (numer1 + numer2 + numer3)*T)*z
        denom1 = ((1 - beta)**2/24)*(np.log(F/X))**2
        denom2 = (((1 - beta)**4)/1920)*((np.log(F/X))**4)
        denom = ((F*X)**((1 - beta)/2))*(1 + denom1 + denom2)*zhi
        sabrsigma = numer/denom

    return sabrsigma

def SABRCall(S, K, r, alpha, beta, rho, nu, T):
    sabr_vol = SABR(S*np.exp(r*T), K, T, alpha, beta, rho, nu)
    return BlackScholesCall(S, K, r, sabr_vol, T)


def SABRPut(S, K, r, alpha, beta, rho, nu, T):
    sabr_vol = SABR(S*np.exp(r*T), K, T, alpha, beta, rho, nu)
    return BlackScholesPut(S, K, r, sabr_vol, T)

def sabrcallintegrand(K, S, r, T, alpha, beta, rho, nu):
    price = SABRCall(S, K, r, alpha, beta, rho, nu, T) * (-2/9 * K**(-5/3) - 1.5 / K**2)
    return price


def sabrputintegrand(K, S, r, T, alpha, beta, rho, nu):
    price = SABRPut(S, K, r, alpha, beta, rho, nu, T) * (-2/9 * K**(-5/3) - 1.5 / K**2)
    return price

def static_replication(S, r, T, alpha, beta, rho, nu):
  F = S * np.exp(r*T)
  I_put = quad(lambda x: sabrputintegrand(x, S, r, T, alpha, beta, rho, nu), 1e-6, F)
  I_call = quad(lambda x: sabrcallintegrand(x, S, r, T, alpha, beta, rho, nu), F, 5000)
  E_var = np.exp(-r*T) * (F**(1/3) + 1.5 * np.log(F) + 10) + I_put[0] + I_call[0]

  return E_var

E_var = static_replication(S, r, T, alpha, beta, rho, nu)
print('The price of the static-replication portfolio is: %.9f' % E_var)



## Contract 2 - under BlackScholes model

sigma = atm_vol_LN
price_of_contract = np.exp(-r*T) * sigma**2 * T
print('The price of the derivative contract using Black-Scholes is: %.9f' % price_of_contract)

## Contract 2 - under Bachelier model

sigma = atm_vol_N
integrand = lambda x: (np.log((S+sigma*np.sqrt(T)*x)/S)*norm.pdf(x))
lower_bound = -S/(sigma*np.sqrt(T)) # S_T/S_0 > 0 so that log(S_T/S_0) is valid
I = quad(integrand, lower_bound, np.inf)
price_of_contract = - 2 * np.exp(-r*T) * I[0]
print('The price of the derivative contract using Bachelier is: %.9f' % price_of_contract)

## Contract 2 - using static replication under SABR model

F = S * np.exp(r*T)
put_integrand = lambda x: (SABRPut(S, x, r, alpha, beta, rho, nu, T) / x**2)
call_integrand = lambda x: (SABRCall(S, x, r, alpha, beta, rho, nu, T) / x**2)
I_put = quad(put_integrand, 0.0, F)
I_call = quad(call_integrand, F, np.inf)
price_of_contract = 2*(I_put[0] + I_call[0])

print('The price of the static-replication portfolio is: %.9f' % price_of_contract)

