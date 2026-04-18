# Stochastic Modelling in Finance
Project Overview :
This project consists of four parts: 
1. Analytical valuation of European options using different models 
2. Calibration of stochastic models to market data 
3. Pricing exotic derivatives through static replication 
4. Simulation and analysis of dynamic hedging strategies 

## Part 1 - Analytical Option Formula 
Considering the following European options: 
- Vanilla Call/Put 
- Digital Cash-or-Nothing Call/Put 
- Digital Asset-or-Nothing Call/Put

We derive and implement pricing formulas for these options using four models: 
1. Black-Scholes Model 
2. Bachelier Model 
3. Black Model 
4. Displaced Diffusion Model

## Part 2 - Model Calibration 
Using the S&P500 (SPX) and SPDR S&P500 ETF (SPY) option prices across 3 maturities and the discount factors (zero rates),
the Displaced Diffusion Model and SABR model were calibrated to plot fitted implied volatility smiles against market implied volatilies 

## Part 3 - Static Replication 
Exotic European derivatives were priced using: 
1. Black-Scholes Model 
2. Bachelier Model 
3. Static Replication using the SABR Model calibrated from Part 2

## Part 4 - Dynamic Hedging Simulation 
Dynamic Hedging strategy for an at-the-money European call option was simulated with various assumptions. 
Steps: 
1. Sell the ATM call option 
2. Dynamically delta hedge based on Black-Scholes model 
3. Hedge N times during the option's life : 
    - N = 21 (daily hedging)
    - N = 84 (more frequent hedging)

Using 50,000 Monte Carlo simulated paths to simulate stock prices and calculate the hedging error (difference between the replicated 
portfolio and the true payoff at maturity).
The histograms of hedging errors for both N=21 and N=84




## License

This project is intended for educational and research purposes only.
