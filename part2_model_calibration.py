"""
Authors: Jia Yu & Lynn
Description: Displaced Diffusion & SABR Model Calibration
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, least_squares
from scipy.interpolate import interp1d
import sys 
import os

data_dir = './data'

df_spx = pd.read_csv(os.path.join(data_dir, 'SPX_options.csv'))
df_spy = pd.read_csv(os.path.join(data_dir, 'SPY_options.csv'))
df_zero_rates = pd.read_csv(os.path.join(data_dir, 'zero_rates_20201201.csv'))

## pre-process parameters
# initial stock price
SPX_S = 3662.45
SPY_S = 366.02

exdate = df_spx['exdate'].unique()
days_to_expiry = [(pd.Timestamp(str(date)) - pd.Timestamp('2020-12-01')).days for date in exdate]

# interpolation of rates for each days to expiry
days_match = np.interp(days_to_expiry, df_zero_rates['days'].values, df_zero_rates['rate'].values)

# create a dictionary mapping each expiration date to its corresponding interpolated rate
params_dict = dict(zip(exdate, days_match))

def preprocess_params(df):
    """
    Pre-process options dataframes 

    Args: 
    df(dataframe): dataframe containing options data

    Returns:
    Pre-processed options data
    """
    df['payoff'] = np.where(df['cp_flag'] == 'C', 'call', 'put')
    df['strike'] = df['strike_price']*0.001
    df['mid_price'] = (df['best_bid'] + df['best_offer'])/2
    return df

df_spx = preprocess_params(df_spx)
df_spy = preprocess_params(df_spy)
    
def calc_black_scholes_d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    sigma_sqrt_time = sigma * np.sqrt(T)
    return ( np.log(S / K) + (r + np.power(sigma, 2)/2) * T ) / sigma_sqrt_time

def calc_black_scholes_d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    sigma_sqrt_time = sigma * np.sqrt(T)
    return calc_black_scholes_d1(S, K, r, sigma, T) - sigma_sqrt_time

def blackscholescall(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_scholes_d1(S, K, r, sigma, T)
    d2 = calc_black_scholes_d2(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return S * norm.cdf(d1)  - K * discount_factor * norm.cdf(d2)

def blackscholesput(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = calc_black_scholes_d1(S, K, r, sigma, T)
    d2 = calc_black_scholes_d2(S, K, r, sigma, T)
    discount_factor = np.exp(-r * T)
    return K * discount_factor * norm.cdf(-d2) - S * norm.cdf(-d1)

def calc_dd_d1(S: float, K: float, r: float, sigma: float, T: float, beta: float):
    F = S * np.exp(r*T)
    Fd = F/beta
    Kd = K + ((1-beta)/beta)*F
    sigma_d = sigma*beta
    return (np.log(Fd/Kd) + (0.5*sigma_d**2)*T)/(sigma_d*np.sqrt(T))

def calc_dd_d2(S: float, K: float, r: float, sigma: float, T: float, beta: float):
    sigma_d = sigma*beta
    return calc_dd_d1(S, K, r, sigma, T, beta) - sigma_d*np.sqrt(T)

def displaceddiffusioncall(S: float, K: float, r: float, sigma: float, T: float, beta: float):
    F = S * np.exp(r*T)
    Fd = F/beta
    Kd = K + ((1-beta)/beta)*F
    d1 = calc_dd_d1(S, K, r, sigma, T, beta)
    d2 = calc_dd_d2(S, K, r, sigma, T, beta)
    return Fd*np.exp(-r*T)*norm.cdf(d1) - Kd*np.exp(-r*T)*norm.cdf(d2)

def displaceddiffusionput(S: float, K: float, r: float, sigma: float, T: float, beta: float):
    F = S * np.exp(r*T)
    Fd = F/beta
    Kd = K + ((1-beta)/beta)*F
    d1 = calc_dd_d1(S, K, r, sigma, T, beta)
    d2 = calc_dd_d2(S, K, r, sigma, T, beta)
    return -Fd*np.exp(-r*T)*norm.cdf(-d1) + Kd*np.exp(-r*T)*norm.cdf(-d2)

def sabrcalibration(x, strikes, vols, F, T):
    err = 0.0
    beta = 0.7
    for i, vol in enumerate(vols):
        err += (vol - SABR(F, strikes[i], T,
                           x[0], beta, x[1], x[2]))**2

    return err

def SABR(F, K, T, alpha, beta, rho, nu):
    X = K
    # if K is at-the-money-forward
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

def impliedvolatility(S: float, K: float, r:float, price: float, T: float, payoff: str):
    """
    Calculate implied volatility based on market price

    Args:
    S(float): spot price
    K(float): strike price
    r(float): risk-free interest rate
    T(float): time to expiration (in years)
    price(float): mid price between bid and offer
    payoff(str): 'call' or 'put'

    Returns:
    Implied Volatility
    """
    try:
        if (payoff.lower() == 'call'):
            impliedVol = brentq(lambda x: price -
                                blackscholescall(S, K, r, x, T),
                                1e-12, 10.0)
        elif (payoff.lower() == 'put'):
            impliedVol = brentq(lambda x: price -
                                blackscholesput(S, K, r, x, T),
                                1e-12, 10.0)
        else:
            raise NameError('Payoff type not recognized')
    except Exception:
        impliedVol = np.nan

    return impliedVol

def dd_calibration(x: list, strikes: float, vols: list, S: float, r: float, sigma: float, T:float, payoff: str):
    """
    To obtain the total squared error between estimated and actual prices

    Args:
    x(list): betas
    strikes(float): strike prices
    vols(list): market implied volatility
    S(float): spot price
    atm_iv(float): atm implied volatility
    T(float): time to expiration (in years)
    payoff(str): 'call' or 'put'

    Returns:
    Total error from list of vols 
    """
    err = 0.0
    for i, vol in enumerate(vols):
        if payoff[i] == 'call':
            dd_price = displaceddiffusioncall(S, strikes[i], r, sigma, T, x[0])
        else:
            dd_price = displaceddiffusionput(S, strikes[i], r, sigma, T, x[0])
        implied_vol = impliedvolatility(S, strikes[i], r, dd_price, T, payoff[i])
        err+=(vol-implied_vol)**2      
    return err  

## Displaced Diffusion Parameters
def calc_model_params(df, S, params_dict, impliedvolatility, dd_calibration):
    model_params = {}
    combined_df_vols = pd.DataFrame()

    for exp_date, days_match in params_dict.items():
        df_exp = df[df['exdate'] == exp_date].copy()
        days_to_expiry = (pd.Timestamp(str(exp_date)) - pd.Timestamp('2020-12-01')).days
        T = days_to_expiry/365
        r = days_match/100
        F = S*np.exp(r*T)

        df_exp.loc[:,'market_vol'] = df_exp.apply(
            lambda x: impliedvolatility(
                S, 
                x['strike'], 
                r, 
                x['mid_price'], 
                T, 
                x['payoff']
                ), 
                axis=1
                )

        df_exp.dropna(inplace=True)
        df_call = df_exp[df_exp['payoff'] == 'call']
        df_put = df_exp[df_exp['payoff'] == 'put']
        strikes = df_put['strike'].values
        
        market_implied_vols = []
        payoff = []
        for K in strikes:
            if K>S:
                payoff.append("call")
                market_implied_vols.append(df_call[df_call['strike'] == K]['market_vol'].values[0])
            else:
                payoff.append("put")
                market_implied_vols.append(df_put[df_put['strike'] == K]['market_vol'].values[0])
        
        df_vols = pd.DataFrame(
            {
                'strike': strikes,
                'market_vol': market_implied_vols,
                'payoff': payoff,
            }
            )
        df_vols['strike_diff'] = np.abs(df_vols['strike']- F)
        atm_min_diff = df_vols.loc[df_vols['strike_diff'].idxmin()]
        atm_iv = atm_min_diff['market_vol']

        beta_initial_guess = [0.3]

        res = least_squares(
            lambda beta: dd_calibration(
                beta, 
                df_vols['strike'],
                df_vols['market_vol'],
                S, 
                r,
                atm_iv,
                T,
                df_vols['payoff'],
            ), 
            beta_initial_guess,
            bounds=(0,1)
        )
        dd_beta = res.x[0]

        model_params[days_to_expiry] = {
            'atm_iv': atm_iv,
            'beta': dd_beta
        } 
        print(model_params)

        dd_implied_vols = []
        for i, K in enumerate(df_vols['strike']):
            payoff = df_vols['payoff'][i]
            if payoff == 'call':
                dd_price = displaceddiffusioncall(S, K, r, atm_iv, T, dd_beta)
            else:
                dd_price = displaceddiffusionput(S, K, r, atm_iv, T, dd_beta)
            dd_vols = impliedvolatility(S, K, r, dd_price, T, payoff)
            dd_implied_vols.append(dd_vols)

        df_vols['dd_implied_vols'] = dd_implied_vols
        df_vols['days_to_expiry'] = days_to_expiry
        combined_df_vols = pd.concat([combined_df_vols, df_vols], ignore_index=True)

    return combined_df_vols

## plot for DD Model SPX options
df_vols_spx = calc_model_params(df_spx, SPX_S, params_dict, impliedvolatility, dd_calibration)
plt.figure(figsize=(10,6))
plt.title('SPX_Options: Implied Volatility vs Strike')

for days_to_expiry in sorted(df_vols_spx['days_to_expiry'].unique()):
    df_vol_unique_spx = df_vols_spx[df_vols_spx['days_to_expiry']==days_to_expiry]
    market_vols_spx = df_vol_unique_spx['market_vol']
    dd_vols_spx = df_vol_unique_spx['dd_implied_vols']
    strikes_spx = df_vol_unique_spx['strike']

    plt.plot(strikes_spx, market_vols_spx, 's', label = f"Market Implied Vol for {days_to_expiry} days")
    plt.plot(strikes_spx, dd_vols_spx, linestyle = '-', label = f"DD Implied Vol for {days_to_expiry} days")

plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.legend()
plt.show()

## plot for DD Model SPY options
df_vols_spy = calc_model_params(df_spy, SPY_S, params_dict, impliedvolatility, dd_calibration)
plt.figure(figsize=(12,8))
plt.title('SPY_Options: Implied Volatility vs Strike')

for days_to_expiry in sorted(df_vols_spy['days_to_expiry'].unique()):
    df_vol_unique_spy = df_vols_spy[df_vols_spy['days_to_expiry']==days_to_expiry]
    market_vols_spy = df_vol_unique_spy['market_vol']
    dd_vols_spy = df_vol_unique_spy['dd_implied_vols']
    strikes_spy = df_vol_unique_spy['strike']

    plt.plot(strikes_spy, market_vols_spy, 's', label = f"Market Implied Vol for {days_to_expiry} days")
    plt.plot(strikes_spy, dd_vols_spy, linestyle = '-', label = f"DD Implied Vol for {days_to_expiry} days")

plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.legend()
plt.show()

## SABR Parameters
def calc_model_params_sabr(df, S, params_dict, impliedvolatility, SABR):
    sabr_model_params = {}
    sabr_combined_df_vols = pd.DataFrame()

    for exp_date, days_match in params_dict.items():
        df_exp = df[df['exdate'] == exp_date].copy()
        days_to_expiry = (pd.Timestamp(str(exp_date)) - pd.Timestamp('2020-12-01')).days
        T = days_to_expiry/365
        r = days_match/100
        F = S*np.exp(r*T)
        beta = 0.7

        df_exp['market_vol'] = df_exp.apply(
            lambda x: impliedvolatility(
                S, 
                x['strike'], 
                r, 
                x['mid_price'], 
                T, 
                x['payoff']
                ), 
                axis=1
                )
        df_exp.dropna(inplace=True)
        df_call = df_exp[df_exp['payoff'] == 'call']
        df_put = df_exp[df_exp['payoff'] == 'put']
        strikes = df_put['strike'].values
        
        market_implied_vols = []
        payoff = []
        for K in strikes:
            if K>S:
                payoff.append("call")
                market_implied_vols.append(df_call[df_call['strike'] == K]['market_vol'].values[0])
            else:
                payoff.append("put")
                market_implied_vols.append(df_put[df_put['strike'] == K]['market_vol'].values[0])
        
        df_sabr_vols = pd.DataFrame(
            {
                'strike': strikes,
                'market_vol': market_implied_vols,
                'payoff': payoff,
            }
            )

        initial_guess = [0.02, 0.2, 0.1]

        res = least_squares(
            lambda x: sabrcalibration(
                x, 
                df_sabr_vols['strike'],
                df_sabr_vols['market_vol'],
                F, 
                T,
            ), 
            initial_guess
        )
        sabr_alpha = res.x[0]
        sabr_rho = res.x[1]
        sabr_nu = res.x[2]

        sabr_model_params[days_to_expiry] = {
            'alpha': sabr_alpha,
            'rho': sabr_rho,
            'nu': sabr_nu
        } 
        print(sabr_model_params)
        sabr_implied_vols = []

        for K in strikes:
            sabr_vols = SABR(F, K, T, sabr_alpha, beta, sabr_rho, sabr_nu)
            sabr_implied_vols.append(sabr_vols)

        df_sabr_vols['sabr_implied_vols'] = sabr_implied_vols
        df_sabr_vols['days_to_expiry'] = days_to_expiry
        sabr_combined_df_vols = pd.concat([sabr_combined_df_vols, df_sabr_vols], ignore_index=True)

    return sabr_combined_df_vols

## plot for SABR Model SPX options
df_sabr_vols_spx = calc_model_params_sabr(df_spx, SPX_S, params_dict, impliedvolatility, SABR)
plt.figure(figsize=(10,6))
plt.title('SPX_Options: Implied Volatility vs Strike')

for days_to_expiry in sorted(df_sabr_vols_spx['days_to_expiry'].unique()):
    df_sabr_vol_unique_spx = df_sabr_vols_spx[df_sabr_vols_spx['days_to_expiry']==days_to_expiry]
    sabr_market_vols_spx = df_sabr_vol_unique_spx['market_vol']
    sabr_vols_spx = df_sabr_vol_unique_spx['sabr_implied_vols']
    sabr_strikes_spx = df_sabr_vol_unique_spx['strike']

    plt.plot(sabr_strikes_spx, sabr_market_vols_spx, 's', label = f"Market Implied Vol for {days_to_expiry} days")
    plt.plot(sabr_strikes_spx, sabr_vols_spx, linestyle = '-', label = f"SABR Implied Vol for {days_to_expiry} days")

plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.legend()
plt.show()

## plot for SABR Model SPY options
df_sabr_vols_spy = calc_model_params_sabr(df_spy, SPY_S, params_dict, impliedvolatility, SABR)
plt.figure(figsize=(10,6))
plt.title('SPY_Options: Implied Volatility vs Strike')

for days_to_expiry in sorted(df_sabr_vols_spy['days_to_expiry'].unique()):
    df_sabr_vol_unique_spy = df_sabr_vols_spy[df_sabr_vols_spy['days_to_expiry']==days_to_expiry]
    sabr_market_vols_spy = df_sabr_vol_unique_spy['market_vol']
    sabr_vols_spy = df_sabr_vol_unique_spy['sabr_implied_vols']
    sabr_strikes_spy = df_sabr_vol_unique_spy['strike']

    plt.plot(sabr_strikes_spy, sabr_market_vols_spy, 's', label = f"Market Implied Vol for {days_to_expiry} days")
    plt.plot(sabr_strikes_spy, sabr_vols_spy, linestyle = '-', label = f"SABR Implied Vol for {days_to_expiry} days")

plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.legend()
plt.show()














    

    

    


