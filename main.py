import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import statsmodels.api as sm
import plotly.express as px

df_daywise = pd.read_csv("day_wise.csv", parse_dates=["Date"])

def plot_figure(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(df['Date'], df['New.cases'], color='blue')
    axes[0].set_title("New Cases Over Time")  
    axes[0].set_xlabel("Date")           
    axes[0].set_ylabel("New Cases")

    axes[1].plot(df['Date'], df['Deaths'], color='red')
    axes[1].set_title("Deaths Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")

    axes[2].plot(df['Date'], df['Recovered'], color='green')
    axes[2].set_title("Recovered Over Time")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recovered")

    plt.tight_layout(pad=3.0)
    plt.show()

def plot_figure_dates(df, start_date, end_date):
    interval = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df_filtered = df.loc[interval]
    plot_figure(df_filtered)


#constants for SIR
I0 = df_daywise.iloc[0]['Active']
R0 = df_daywise.iloc[0]['Recovered']
D0 = df_daywise.iloc[0]['Deaths']
S0 = 17000000
N = S0 + I0 + R0 + D0

#parameter set 1
#using goverment data (CDC) for parameters
beta_hat1 = 0.25
gamma_hat1 = 0.1
mu_hat1 = 0.002
alpha_hat1 = 0.0111
params1 = [alpha_hat1, beta_hat1, gamma_hat1, mu_hat1]

#parameter set 2
#using estimators for the parameters:
#taking averages instead of values dependent on t to avoid overfitting
delta_D = df_daywise['Deaths'].diff().iloc[1:]
mu_hat2 = delta_D.div(df_daywise['Active'].iloc[1:]).mean()                                                                      # mu = 0.003
gamma_hat2 = df_daywise['Recovered'].diff().iloc[1:].div(df_daywise['Active'].iloc[1:]).mean()                                      # gamma = 0.02
alpha_hat2 = ((gamma_hat2 * df_daywise['Active'].iloc[1:] - df_daywise['Recovered'].diff().iloc[1:]) / df_daywise['Recovered'].iloc[1:]).mean()    # alpha = 0.055

df_daywise['S'] = N - df_daywise['Active'] - df_daywise['Recovered'] - df_daywise['Deaths']
beta_hat2 = ((N / df_daywise['S'].iloc[1:]) * (df_daywise['Active'].diff().iloc[1:] / df_daywise['Active'].iloc[1:] + mu_hat2 + gamma_hat2)).mean()  # beta = 0.12
params2 = [alpha_hat2, beta_hat2, gamma_hat2, mu_hat2]


#parameter set 3
# Estimate mu, gamma using linear regression on ΔD(t) = μ I(t), and ΔR(t) = γ I(t) respectively      
I_mu = df_daywise['Active'].iloc[1:]                 
model_mu = sm.OLS(delta_D, I_mu)
results_mu = model_mu.fit()
mu_hat3 = results_mu.params.iloc[0]  #mu = 0.0012

delta_R = df_daywise['Recovered'].diff().dropna()   
I_gamma = df_daywise['Active'].iloc[1:]              
model_gamma = sm.OLS(delta_R, I_gamma)
results_gamma = model_gamma.fit()
gamma_hat3 = gamma_hat3 = results_gamma.params.iloc[0] #gamma = 0.025

# Now, estimate α via least squares from:

#α R(t) = γ I(t) - ΔR(t)
I_vals = df_daywise['Active'].iloc[1:].values       
R_vals = df_daywise['Recovered'].iloc[1:].values        
Delta_R = df_daywise['Recovered'].diff().iloc[1:].values 
alpha_hat3 = np.sum(R_vals * (gamma_hat3 * I_vals - Delta_R)) / np.sum(R_vals**2) #alpha = -.0008

#β = [ΔI(t) + (μ+γ) I(t)] / [S(t) I(t)/N]
S_vals = (N - (df_daywise['Active'] + df_daywise['Recovered'] + df_daywise['Deaths'])).iloc[1:].values
I_vals_beta = df_daywise['Active'].iloc[1:].values
Delta_I = df_daywise['Active'].diff().iloc[1:].values
predictor = S_vals * I_vals_beta / N
response = Delta_I + (mu_hat3 + gamma_hat3) * I_vals_beta
beta_hat3 = np.sum(predictor * response) / np.sum(predictor**2) #beta = 0.077
params3 = [alpha_hat3, beta_hat3, gamma_hat3, mu_hat3]


def sir_model_simulation(df, alpha, beta, gamma, mu, I0, R0, S0, D0, N, parameterset):
    S = [S0]
    I = [I0]
    R = [R0]
    D = [D0]
    time = df['Date'].tolist()
    
    for t in range(len(time) - 1):
        St = S[-1]
        It = I[-1]
        Rt = R[-1]
        Dt = D[-1]
        
        delta_S = alpha * Rt - beta * St * It / N
        delta_I = beta * St * It / N - (mu + gamma) * It
        delta_R = gamma * It - alpha * Rt
        delta_D = mu * It
        
        S.append(St + delta_S)
        I.append(It + delta_I)
        R.append(Rt + delta_R)
        D.append(Dt + delta_D)

    S_sim = np.array(S)
    I_sim = np.array(I)
    R_sim = np.array(R)
    D_sim = np.array(D)
    
    S_real = np.array(N - df['Active'] - df['Recovered'] - df['Deaths'])
    I_real = np.array(df['Active'])
    R_real = np.array(df['Recovered'])
    D_real = np.array(df['Deaths'])
    
    mse_S = np.mean((S_sim - S_real) ** 2)
    mse_I = np.mean((I_sim - I_real) ** 2)
    mse_R = np.mean((R_sim - R_real) ** 2)
    mse_D = np.mean((D_sim - D_real) ** 2)
    
    print("Mean Squared Errors for parameterset ", parameterset, ", offset by 1/10000000000000:")
    print("S: ", mse_S / 10000000000000)
    print("I: ", mse_I / 10000000000000)
    print("R: ", mse_R / 10000000000000)
    print("D: ", mse_D / 10000000000000)
    print("Estimate for basic reproduction number R0: ",beta/gamma )
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, S, label="Susceptible")
    plt.plot(time, I, label="Infected")
    plt.plot(time, R, label="Recovered")
    plt.plot(time, D, label="Deceased")
    plt.xlabel("Date")
    plt.ylabel("Number of individuals")
    plt.title(f"SIR Model with Deaths Simulation using parameterset: {parameterset}")
    plt.yscale('log')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    #Part 1
    plot_figure(df_daywise)

    start_date = "2020-03-01"
    end_date = "2020-05-01"
    plot_figure_dates(df_daywise, start_date, end_date)


    parameter_sets = [params1, params2, params3]
    for i, params in enumerate(parameter_sets, start=1):
        sir_model_simulation(df_daywise, params[0], params[1], params[2], params[3],
                                I0, R0, S0, D0, N, i)


main()
