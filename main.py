
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import statsmodels.api as sm
import plotly.express as px
import streamlit as st
import sys

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide", initial_sidebar_state="expanded")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                        #Part 1
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

df_daywise = pd.read_csv("day_wise.csv", parse_dates=["Date"])

def plot_figure(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(df["Date"], df["New cases"], color="blue")
    axes[0].set_title("New Cases Over Time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("New Cases")
    plt.tight_layout(pad=3.0)

    axes[1].plot(df["Date"], df["Deaths"], color="red")
    axes[1].set_title("Deaths Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths")
    plt.tight_layout(pad=3.0)

    axes[2].plot(df["Date"], df["Recovered"], color="green")
    axes[2].set_title("Recovered Over Time")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recovered")
    plt.tight_layout(pad=3.0)

    return fig

def plot_figure_dates(df, start_date, end_date):
    interval = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
    df_filtered = df.loc[interval]
    return plot_figure(df_filtered)

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

parameter_sets = [params1, params2, params3]

def sir_model_MSE_values(df, alpha, beta, gamma, mu, I0, R0, S0, D0, N):
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
    
    # Actual values (assuming your actual DataFrame df has cumulative counts)
    S_real = np.array(N - df['Active'] - df['Recovered'] - df['Deaths'])
    I_real = np.array(df['Active'])
    R_real = np.array(df['Recovered'])
    D_real = np.array(df['Deaths'])
    
    mse_S = np.mean((S_sim - S_real) ** 2)
    mse_I = np.mean((I_sim - I_real) ** 2)
    mse_R = np.mean((R_sim - R_real) ** 2)
    mse_D = np.mean((D_sim - D_real) ** 2)
    
    return mse_S, mse_I, mse_R, mse_D

def plot_mse_comparison(df, parameter_sets, I0, R0, S0, D0, N, param_set_number=None, scale=1e13):
    """
    For each parameter set in parameter_sets (list of [alpha, beta, gamma, mu]),
    compute the MSE values (for S, I, R, D) using original df and the given initial conditions.
    Then create a figure with one barplot per parameter set.
    """
    compartments = ['S', 'I', 'R', 'D']
    mse_results = []
    for i, params in enumerate(parameter_sets, start=1):
        alpha_val = params[0]
        beta_val = params[1]
        gamma_val = params[2]
        mu_val = params[3]
        mse_S, mse_I, mse_R, mse_D = sir_model_MSE_values(df, alpha_val, beta_val, gamma_val, mu_val, I0, R0, S0, D0, N)
        mse_results.append([mse_S/scale, mse_I/scale, mse_R/scale, mse_D/scale])
    
    # Create a figure with one subplot per parameter set
    fig, axes = plt.subplots(1, len(parameter_sets), figsize=(5 * len(parameter_sets), 5))
    if len(parameter_sets) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.bar(compartments, mse_results[i], color=['blue', 'orange', 'green', 'red'])
        if param_set_number:
            ax.set_title(f"Parameter Set {param_set_number} MSE")
        else:
            ax.set_title(f"Parameter Set {i + 1} MSE")
        ax.set_ylim(0, 3.5)
        ax.set_ylabel("MSE (scaled)")
    fig.tight_layout()
    return fig

def graph_sir_model_simulation(df, alpha, beta, gamma, mu, I0, R0, S0, D0, N, parameterset):
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

        
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time, S, label="Susceptible")
    plt.plot(time, I, label="Infected")
    plt.plot(time, R, label="Recovered")
    plt.plot(time, D, label="Deceased")
    plt.xlabel("Date")
    plt.ylabel("Number of individuals")
    plt.title(f"SIR Model using parameterset: {parameterset}")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    return fig


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                        #Part 3
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Loading the database
conn = sqlite3.connect('covid_database.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
table_names = [table[0] for table in tables]

dataframes = {}
for table_name in table_names:
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    dataframes[table_name] = df

conn.close()

country_wise_df = dataframes.get('country_wise')
day_wise_df = dataframes.get('day_wise')
day_wise_df['Date'] = pd.to_datetime(day_wise_df['Date'])
day_wise_df.sort_values("Date", inplace=True)
usa_county_wise_df = dataframes.get('usa_county_wise')
worldometer_df = dataframes.get('worldometer_data')


def plot_totals_for_country(country, start_date, end_date):
    df_pop = worldometer_df[worldometer_df['Country.Region'] == country]
    if df_pop.empty:
        print(f"No population data found for {country}")
        return
    population = df_pop.iloc[0]['Population']
    
    df_plot = day_wise_df.copy()
    interval = (df_plot['Date'] >= pd.to_datetime(start_date)) & (df_plot['Date'] <= pd.to_datetime(end_date))
    df_filtered = df_plot.loc[interval].copy()

    # Compute cumulative totals for each variable
    df_filtered['Active_total'] = df_filtered['Active'].cumsum()
    df_filtered['Deaths_total'] = df_filtered['Deaths'].cumsum()
    df_filtered['Recovered_total'] = df_filtered['Recovered'].cumsum()
    
    # Calculate fractions using the cumulative totals
    df_filtered['Active_fraction'] = df_filtered['Active_total'] / population
    df_filtered['Deaths_fraction'] = df_filtered['Deaths_total'] / population
    df_filtered['Recovered_fraction'] = df_filtered['Recovered_total'] / population
    

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(df_filtered['Date'], df_filtered['Active_fraction'], label=f"Cumulative Active / Population, {country}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Active Fraction (Total)")
    axes[0].legend()
    axes[0].set_title(f"Active Fraction Over Time for {country}")
    plt.tight_layout(pad = 5.0)

    axes[1].plot(df_filtered['Date'], df_filtered['Deaths_fraction'], label=f"Cumulative Deaths / Population, {country}", color='red')
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths Fraction (Total)")
    axes[1].legend()
    axes[1].set_title(f"Deaths Fraction Over Time for {country}")
    plt.tight_layout(pad = 5.0)

    axes[2].plot(df_filtered['Date'], df_filtered['Recovered_fraction'], label=f"Cumulative Recovered / Population, {country}", color='green')
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recovered Fraction (Total)")
    axes[2].legend()
    axes[2].set_title(f"Recovered Fraction Over Time for {country}")
    plt.tight_layout(pad = 5.0)
    
    return fig

def estimate_parameters_by_country(country):
    population = worldometer_df[worldometer_df['Country.Region'] == country]['Population'].iloc[0]

    deaths = day_wise_df['Deaths'] / population
    recovered = day_wise_df['Recovered'] / population
    infected = day_wise_df['Active'] / population

    delta_deaths = deaths.diff()
    delta_recovered = recovered.diff()
    delta_infected = infected.diff()
    
    # Normalized susceptible fraction (time series)
    S_t = 1 - (infected + recovered + deaths)

    gamma = 1 / 4.5

    # Ignore the first value (index 0) which is NaN due to diff()
    mu_t = delta_deaths.iloc[1:] / infected.iloc[1:]
    alpha_t = (gamma * infected.iloc[1:] - delta_recovered.iloc[1:]) / delta_recovered.iloc[1:]
    beta_t = (delta_infected.iloc[1:] / infected.iloc[1:] + mu_t + gamma) / S_t.iloc[1:]

    params = [alpha_t, beta_t, gamma, mu_t, S_t]
    return params

def R0_trajectory_by_country(country):
    # Unpack the parameters:
    # params = [alpha_t, beta_t, gamma, mu_t, S_t]
    alpha_t, beta_t, gamma, mu_t, S_t = estimate_parameters_by_country(country)
    R0_trajectory = beta_t / gamma

    # Since we used .diff(), the series starts at index 1.
    df_result = pd.DataFrame({
         'Date': day_wise_df['Date'].iloc[1:],
         'R0': R0_trajectory.values
    })
    
    return df_result

def Active_Cases_fraction_Europe():
    df_europe = worldometer_df[worldometer_df['Continent'] == 'Europe'].copy()
    df_europe['ActiveFraction'] = (df_europe['ActiveCases'] / df_europe['Population']).round(4)

    fig = px.choropleth(
        df_europe,
        locations="Country.Region",               
        locationmode="country names",      
        color="ActiveFraction",            
        scope="europe",                    
        title="Active Cases per habitant in Europe",
        color_continuous_scale="Reds"      
    )

    return fig

def Estimated_Death_Rate_by_Continent():
    worldometer_df['Continent'] = worldometer_df['Continent'].replace("", "Antartica")
    worldometer_df['DeathRate'] = worldometer_df['TotalDeaths'] / worldometer_df['TotalCases']
    death_rate_by_continent = worldometer_df.groupby('Continent')['DeathRate'].mean().reset_index()

    fig = plt.figure(figsize=(10, 6))
    plt.bar(death_rate_by_continent['Continent'], death_rate_by_continent['DeathRate'], color='salmon') 
    plt.xlabel('Continent')
    plt.ylabel('Average Death Rate (TotalDeaths/TotalCases)')
    plt.title('Average Death Rate by Continent')
    plt.tight_layout()
    
    return fig

def Top_5_US_Counties():
    county_summary = usa_county_wise_df.groupby('Admin2').agg({'Deaths': 'sum', 'Confirmed': 'sum'}).reset_index()
    top5_deaths = county_summary.sort_values(by='Deaths', ascending=False).head(5)
    top5_cases = county_summary.sort_values(by='Confirmed', ascending=False).head(5)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Top 5 US Counties by Deaths
    axes[0].bar(top5_deaths['Admin2'], top5_deaths['Deaths'], color='red')
    axes[0].set_xlabel('County')
    axes[0].set_ylabel('Total Deaths')
    axes[0].set_title('Top 5 US Counties by Total Deaths')
    axes[0].tick_params(axis='x', rotation=45)

    # Top 5 US Counties by Cases
    axes[1].bar(top5_cases['Admin2'], top5_cases['Confirmed'], color='blue')
    axes[1].set_xlabel('County')
    axes[1].set_ylabel('Total Cases')
    axes[1].set_title('Top 5 US Counties by Total Cases')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout(pad=5.0)
    return fig

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                        #Part 4
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Use interpolation to fill in missing values
def manual_interpolate_column(series):
    # Manually interpolate a pandas Series.
    s = series.copy()
    for i in range(len(s)):
        if pd.isna(s.iloc[i]):
            if i == 0:
                s.iloc[i] = 0
            else:
                # Find the last non-missing value before i
                j = i - 1
                while j >= 0 and pd.isna(s.iloc[j]):
                    j -= 1
                # If none found, use 0.
                if j < 0:
                    s.iloc[i] = 0
                    continue
                # Find the next non-missing value after i
                k = i + 1
                while k < len(s) and pd.isna(s.iloc[k]):
                    k += 1
                if k < len(s):
                    # Linear interpolation:
                    # s[i] = s[j] + (s[k] - s[j]) * ((i - j) / (k - j))
                    s.iloc[i] = s.iloc[j] + (s.iloc[k] - s.iloc[j]) * ((i - j) / (k - j))
                else:
                    # If no next non-missing value, fill with the last known value.
                    s.iloc[i] = s.iloc[j]
    return s

def fill_single_missing(row):
    # If exactly one missing value among Confirmed, Active, Deaths, and Recovered,
    # fill it using the invariant.
    if row[['Confirmed', 'Active', 'Deaths', 'Recovered']].isna().sum() == 1:
        if pd.isna(row['Active']):
            row['Active'] = row['Confirmed'] - row['Deaths'] - row['Recovered']
        elif pd.isna(row['Deaths']):
            row['Deaths'] = row['Confirmed'] - row['Active'] - row['Recovered']
        elif pd.isna(row['Recovered']):
            row['Recovered'] = row['Confirmed'] - row['Active'] - row['Deaths']
        elif pd.isna(row['Confirmed']):
            row['Confirmed'] = row['Active'] + row['Deaths'] + row['Recovered']
    return row

def fill_row_manual(row, df_interp):
    # We only manually interpolate for Confirmed, Deaths, and Recovered as they are cumulative and less prone to errors (only positive values for instance)
    cols = ['Confirmed', 'Deaths', 'Recovered']
    if row[cols].isna().sum() >= 2:
        # Fill each missing value from our manually interpolated reference.
        for col in cols:
            if pd.isna(row[col]):
                row[col] = df_interp.at[row.name, col]
        # Now if exactly one missing remains, fill it using the invariant.
        if row[['Confirmed', 'Active', 'Deaths', 'Recovered']].isna().sum() == 1:
            row = fill_single_missing(row)
    return row

# function to handle the raw data processing
def process_country_complete(country):
    complete_df = pd.read_csv("complete.csv", parse_dates=["Date"])
    
    df = complete_df[complete_df['Country.Region'] == country][
        ['Date', 'Confirmed', 'Active', 'Deaths', 'Recovered']
    ].copy()
    df.sort_values("Date", inplace=True)

    df = df.groupby('Date', as_index=False).agg({
    'Confirmed': 'max',
    'Active': 'max',
    'Deaths': 'max',
    'Recovered': 'max'
    })

    # Drop initial rows where all four key columns are missing.
    valid = ~df[['Confirmed', 'Active', 'Deaths', 'Recovered']].isna().all(axis=1)
    if valid.any():
        first_valid = df.index[valid][0]
        df = df.loc[first_valid:]
        start_date = df.iloc[0]["Date"]
    else:
        print(f"No valid rows for {country}")
        return df
    
     # Special handling:
    # If Confirmed equals Active and both Deaths and Recovered are missing,
    # then set Deaths and Recovered to 0.
    condition = (df['Confirmed'] == df['Active']) & (df['Deaths'].isna()) & (df['Recovered'].isna())
    df.loc[condition, ['Deaths', 'Recovered']] = 0

    # First, for rows with exactly one missing value, fill using the invariant.
    df = df.apply(fill_single_missing, axis=1)
    
    # Create an interpolated reference for Confirmed, Deaths, and Recovered using manual interpolation.
    df_interp = df.copy()
    for col in ['Confirmed', 'Deaths', 'Recovered']:
        df_interp[col] = manual_interpolate_column(df_interp[col])
    
    # For rows with two or more missing among these columns, fill using our manual interpolation.
    df = df.apply(lambda row: fill_row_manual(row, df_interp) if row[['Confirmed', 'Deaths', 'Recovered']].isna().sum() >= 2 else row, axis=1)
    
    # Finally, if Active is missing but the others are filled, compute Active using the invariant.
    missing_active = df['Active'].isna()
    df.loc[missing_active, 'Active'] = df.loc[missing_active, 'Confirmed'] - df.loc[missing_active, 'Deaths'] - df.loc[missing_active, 'Recovered']
    
    # Drop any rows still containing missing values.
    df_complete = df.dropna(subset=['Confirmed', 'Active', 'Deaths', 'Recovered'])

    return df_complete

# Plots totals
def plot_figures_country_complete(country, start_date, end_date):
    df = process_country_complete(country).copy()

    interval = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    df_filtered = df.loc[interval].copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    axes[0].plot(df_filtered["Date"], df_filtered["Confirmed"], color="purple")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cumulative Confirmed")
    axes[0].set_title(f"Cumulative Confirmed for {country}")
    plt.tight_layout(pad=5.0)
    
    axes[1].plot(df_filtered["Date"], df_filtered["Active"], color="blue")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Active")
    axes[1].set_title(f"Cumulative Active for {country}")
    plt.tight_layout(pad=5.0)
    
    axes[2].plot(df_filtered["Date"], df_filtered["Deaths"], color="red")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Cumulative Deaths")
    axes[2].set_title(f"Cumulative Deaths for {country}")
    plt.tight_layout(pad=5.0)
    
    axes[3].plot(df_filtered["Date"], df_filtered["Recovered"], color="green")
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel("Cumulative Recovered")
    axes[3].set_title(f"Cumulative Recovered for {country}")
    plt.tight_layout(pad=5.0)
    
    return fig

# Estimating params for the SIR model; use epsilon (eps) instead of dividing by 0
def estimates_country_complete(country):
    df = process_country_complete(country).copy()

    population = worldometer_df[worldometer_df['Country.Region'] == country]['Population'].iloc[0]
    df['Confirmed_change'] = df['Confirmed'].diff()
    df['Active_change'] = df['Active'].diff()
    df['Deaths_change'] = df['Deaths'].diff()
    df['Recovered_change'] = df['Recovered'].diff()
    
    S_t =  population - (df['Active'] + df['Recovered'] + df['Deaths'])
    gamma = 1 / 4.5

    # Ignore the first value (index 0) which is NaN due to diff()
    eps = 0.00001
    mu_t = df['Deaths_change'].iloc[1:] / np.maximum(df['Active'].iloc[1:], eps)
    alpha_t = (gamma * df['Active'].iloc[1:] - df['Recovered_change'].iloc[1:]) / np.maximum(df['Recovered_change'].iloc[1:], eps)
    beta_t = (df['Active_change'].iloc[1:] / np.maximum(df['Active'].iloc[1:], 1) + mu_t + gamma) / np.maximum(S_t.iloc[1:], eps)
    R0 = beta_t / gamma

    params = [alpha_t, beta_t, gamma, mu_t, S_t, R0]
    return params

def plot_figures_counties_complete(county):
    county_df = usa_county_wise_df[usa_county_wise_df['Admin2'] == county].copy()
    
    county_df['Date'] = pd.to_datetime(county_df['Date'], format='%m/%d/%y')
    county_df.sort_values("Date", inplace=True)
    
    fig = plt.figure(figsize=(10, 16))
    
    plt.subplot(2, 1, 1)
    plt.plot(county_df['Date'], county_df['Confirmed'], color='purple')
    plt.xlabel("Date")
    plt.ylabel("Confirmed Change")
    plt.title(f"Daily Change in Confirmed Cases for {county}")
    
    plt.subplot(2, 1, 2)
    plt.plot(county_df['Date'], county_df['Deaths'], label="Daily Change in Deaths", color='blue')
    plt.xlabel("Date")
    plt.ylabel("Deaths Change")
    plt.title(f"Daily Change in Deaths for {county}")

    plt.tight_layout(pad = 5.0)
    return fig

def main():
    # Part 1
    plot_figure(df_daywise)
    plt.show()

    start_date = "2020-03-01"
    end_date = "2020-05-01"
    plot_figure_dates(df_daywise, start_date, end_date)
    plt.show()

    for i, params in enumerate(parameter_sets, start=1):
        graph_sir_model_simulation(df_daywise, params[0], params[1], params[2], params[3],
                                I0, R0, S0, D0, N, i)
        plt.show()
    plot_mse_comparison(df_daywise, parameter_sets, I0, R0, S0, D0, N)
    plt.show()

    # Part 3
    start_date = "2020-01-22"
    end_date = "2020-07-27"
    plot_totals_for_country("Netherlands", start_date, end_date)
    plt.show()

    Active_Cases_fraction_Europe().show()
    Estimated_Death_Rate_by_Continent()
    plt.show()

    Top_5_US_Counties()
    plt.show()

    # Part 4
    start_date = "2020-01-22"
    end_date = "2020-07-27"
    plot_figures_country_complete("Netherlands", start_date, end_date)
    plt.show()
    plot_figures_counties_complete("Hudson")
    plt.show()



if "run_main" in sys.argv:
    main()


#Part 5
#Streamlit Dashboard Configuration

# Test to see accuracy of SIR model
def test_SIR_Model(param_country, sim_country):
    """
    Estimate time-dependent SIR parameters from param_country, then simulate the epidemic
    for sim_country using those daily parameter estimates. Plot simulated vs. actual curves
    for Confirmed, Active, and Deaths over time.
    """
    # --- Step 1: Estimate parameters for param_country ---
    params = estimates_country_complete(param_country)
    alpha_series = params[0]
    beta_series  = params[1]
    gamma        = params[2]
    mu_series    = params[3]
    
     # --- Step 2: Get actual epidemic data for sim_country ---
    actual_df = process_country_complete(sim_country).copy()
    # Determine the available number of rows.
    n_actual = len(actual_df)
    # Note: estimates_country_complete uses .diff() so its series length is (len(df) - 1)
    n_alpha = len(alpha_series)
    # The simulation will run for n_steps days, where initial condition uses one extra row.
    n_steps = min(n_alpha, n_actual - 1)
    
    # Slice the actual_df so that we have n_steps+1 rows (initial condition plus n_steps steps)
    actual_df = actual_df.iloc[-(n_steps + 1):].reset_index(drop=True)
    t_dates = actual_df["Date"].tolist()

    # --- Step 3: Extract initial conditions from sim_country's data (first available day) ---
    init_row = actual_df.iloc[0]

    # Assume total population for simulation is taken as the first day's population from worldometer data
    N_sim = worldometer_df[worldometer_df['Country.Region'] == sim_country]['Population'].iloc[0] 
    
    # --- Step 4: Run SIR simulation with time-dependent parameters ---
    I_sim = [init_row["Active"]]   # Active is considered as Infected
    R_sim = [init_row["Recovered"]]
    D_sim = [init_row["Deaths"]]
    S_sim = [N_sim - I_sim[0] - R_sim[0] - D_sim[0]]

    # Simulate for n_steps days.
    for t in range(n_steps):
        alpha = alpha_series.iloc[t]
        beta  = beta_series.iloc[t]
        mu    = mu_series.iloc[t]

        St = S_sim[-1]
        It = I_sim[-1]
        Rt = R_sim[-1]
        Dt = D_sim[-1]
        
        # SIR model with deaths equations (Euler method, time step = 1 day)
        delta_S = alpha * Rt - (beta * St * It / N_sim)
        delta_I = beta * St * It / N_sim - (mu + gamma) * It
        delta_R = gamma * It - alpha * Rt
        delta_D = mu * It

        S_sim.append(St + delta_S)
        I_sim.append(It + delta_I)
        R_sim.append(Rt + delta_R)
        D_sim.append(Dt + delta_D)

    S_sim = np.array(S_sim)
    I_sim = np.array(I_sim)
    R_sim = np.array(R_sim)
    D_sim = np.array(D_sim)
    
    # --- Step 5: Plot simulated vs. actual data ---
    # Compute actual confirmed from actual data (Confirmed = Active + Deaths + Recovered)
    actual_confirmed = actual_df["Active"] + actual_df["Deaths"] + actual_df["Recovered"]
    sim_confirmed = I_sim + R_sim + D_sim
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Confirmed Cases
    axes[0].plot(t_dates, actual_confirmed, label="Actual Confirmed", marker="o", color="purple")
    axes[0].plot(t_dates, sim_confirmed, label="Simulated Confirmed", marker="x", linestyle="--", color="orange")
    axes[0].set_title(f"Confirmed Cases Comparison for {param_country} and {sim_country}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cumulative Confirmed")
    axes[0].legend()
    
    # Active Cases
    axes[1].plot(t_dates, actual_df["Active"], label="Actual Active", marker="o", color="blue")
    axes[1].plot(t_dates, I_sim, label="Simulated Active", marker="x", linestyle="--", color="green")
    axes[1].set_title(f"Active Cases Comparison for {param_country} and {sim_country}")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Active Cases")
    axes[1].legend()
    
    # Deaths
    axes[2].plot(t_dates, actual_df["Deaths"], label="Actual Deaths", marker="o", color="red")
    axes[2].plot(t_dates, D_sim, label="Simulated Deaths", marker="x", linestyle="--", color="black")
    axes[2].set_title(f"Deaths Comparison for {param_country} and {sim_country}")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Cumulative Deaths")
    axes[2].legend()
    
    plt.tight_layout(pad=5.0)
    return fig


def test_SIR_Model_R0_trajectory(param_country, sim_country):
    """
    Estimate time-dependent SIR parameters for two countries and plot their R₀ trajectories in a single figure.
    
    Parameters:
    - param_country: The country from which the SIR parameters are estimated.
    - sim_country: The country for which the R₀ trajectory is compared to.
    
    Returns:
    - A matplotlib figure showing the R₀ trajectories for both countries.
    """
    # --- Step 1: Estimate parameters and R₀ for param_country ---
    params_param = estimates_country_complete(param_country)
    R0_param = params_param[5]
    # Retrieve the corresponding dates (skipping the first row due to diff())
    df_param = process_country_complete(param_country).copy()
    dates_param = df_param["Date"].iloc[1:].reset_index(drop=True)
    
    # --- Step 2: Estimate parameters and R₀ for sim_country ---
    params_sim = estimates_country_complete(sim_country)
    R0_sim = params_sim[5]
    df_sim = process_country_complete(sim_country).copy()
    dates_sim = df_sim["Date"].iloc[1:].reset_index(drop=True)
    
    # --- Step 3: Create the figure with both R₀ trajectories ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates_param, R0_param, label=f"R₀ for {param_country}", marker="o", color="blue")
    ax.plot(dates_sim, R0_sim, label=f"R₀ for {sim_country}", marker="x", linestyle="--", color="red")
    
    ax.set_title(f"R₀ Trajectory Comparison: {param_country} vs. {sim_country}")
    ax.set_xlabel("Date")
    ax.set_ylabel("R₀")
    ax.legend()
    plt.tight_layout(pad=5.0)
    
    return fig


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "R₀ Trajectory", "SIR Model Parameter Comparison", "SIR Model Fit test", "Country Analysis", "Global Insights", "Counties Analysis"])

if page in ["Overview", "Country Analysis"]:
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [df_daywise["Date"].min(), df_daywise["Date"].max()]
    )

if page in ["R₀ Trajectory", "Country Analysis"]:
    selected_country = st.sidebar.selectbox("Select a country", worldometer_df["Country.Region"].unique())


if page == "Overview":
    st.title("COVID-19 Pandemic Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Statistics")
        total_cases = worldometer_df["TotalCases"].sum()
        total_deaths = worldometer_df["TotalDeaths"].sum()
        total_recovered = worldometer_df["TotalRecovered"].sum()
        st.metric("Total Cases", f"{total_cases:,}")
        st.metric("Total Deaths", f"{total_deaths:,}")
        st.metric("Total Recovered", f"{total_recovered:,}")
    with col2:
        st.subheader("Global Insights")
        fig = plot_figure_dates(df_daywise, start_date, end_date)
        st.pyplot(fig)

elif page == "R₀ Trajectory":
    st.title("R₀ Trajectory for COVID-19 Spread")
    st.markdown("""
    The basic reproduction number (R₀) represents the average number of secondary infections caused by one infected individual in a fully susceptible population.
    
    An R₀ value:
    - Greater than 1 means the disease is spreading.
    - Equal to 1 means the disease is stable.
    - Less than 1 means the disease is declining.

    The R₀ trajectory shows how the transmission potential changes over time, often influenced by public health interventions, immunity buildup, and other factors.
    """)

    # Get the R₀ trajectory data for the selected country
    R0_traj_df = R0_trajectory_by_country(selected_country)

    # Plot R₀ trajectory
    st.subheader(f"R₀ Trajectory for {selected_country}")
    fig_R0 = plt.figure(figsize=(10, 5))
    plt.plot(R0_traj_df["Date"], R0_traj_df["R0"], label="R₀ Estimate", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Estimated R₀")
    plt.title(f"R₀ Trajectory for {selected_country}")
    plt.axhline(y=1, color='red', linestyle='--', label='Threshold (R₀ = 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig_R0)

elif page == "SIR Model Parameter Comparison":
    st.title("SIR Model Parameter Set Comparison")
    st.markdown("""
    The SIR model is used to predict the spread of infectious diseases by categorizing people into:
    - **S (Susceptible):** People who can still get infected.
    - **I (Infected):** People who are currently infected.
    - **R (Recovered):** People who have recovered and gained immunity.
    - **D (Deceased):** People who have died from the disease.

    The model is influenced by four parameters:
    - **α (alpha):** Rate at which recovered people lose immunity and become susceptible again.
    - **β (beta):** Infection rate (how easily the disease spreads).
    - **γ (gamma):** Recovery rate (how quickly people recover).
    - **μ (mu):** Death rate due to infection.
        
    In this section, you can select one of the three parameter sets and visualize the SIR model simulation along with the Mean Squared Error (MSE) for that set.

    - **Parameter Set 1:** Using government data (CDC) for the parameters.
    - **Parameter Set 2:** Calculating the parameters from the data and taking a mean to avoid overfitting.
    - **Parameter Set 3:** Using advanced techniques such as linear regression to compute parameters.
    """)

    # Dropdown menu to select the parameter set
    parameter_set_choice = st.selectbox("Select Parameter Set", ["Parameter Set 1", "Parameter Set 2", "Parameter Set 3"])

    # Mapping dropdown selection to actual parameter set
    param_index = int(parameter_set_choice[-1]) - 1
    selected_params = parameter_sets[param_index]

    # Display the selected parameter set's simulation
    fig_sim = graph_sir_model_simulation(df_daywise, selected_params[0], selected_params[1], selected_params[2], selected_params[3],
                                         I0, R0, S0, D0, N, param_index + 1)
    st.pyplot(fig_sim)

    # Display the MSE comparison for the selected parameter set
    fig_mse = plot_mse_comparison(df_daywise, [selected_params], I0, R0, S0, D0, N)
    st.pyplot(fig_mse)

elif page == "SIR Model Fit test":
    st.title("SIR Model Fit Test")
    st.markdown("""
    This section estimates SIR model parameters from one country (e.g., the Netherlands) and then uses those parameters 
    to simulate the epidemic for another country that is comparable (e.g., Belgium) using the starting values from that country.
    The simulation outputs the modeled cumulative confirmed, active, death cases, and R0 trajectory. If the computed results are similar to the actual results,
    the SIR model proves a fitting model to predict said statistics.

    We conclude the model does not predict figures such as the confirmed cases well. More explained in the readme file.           
    """)
    # Adjust the country names as needed; here we use "Netherlands" for estimation and "Belgium" for simulation.
    fig_sim = test_SIR_Model("Netherlands", "Belgium")
    st.pyplot(fig_sim)
    st.markdown("""
    The simulated R0-trajectory is similar enough to that of the actual R0-trajectory to conlude we can use the model for this purpose.         
    """)
    st.pyplot(test_SIR_Model_R0_trajectory("Netherlands", "Belgium"))

elif page == "Country Analysis":
    complete_csv_country = "US" if selected_country == "USA" else selected_country
    st.title(f"COVID-19 Analysis for {selected_country}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cumulative Totals")
        st.markdown(
             """
             This plot displays the cumulative totals of active cases, deaths, and recoveries over time from day_wise data file,
             normalized by the country's population.
             """
         )
        fig1 = plot_totals_for_country(selected_country, start_date, end_date)
        fig1.tight_layout()
        st.pyplot(fig1)
    with col2:
        st.subheader("Complete Data Analysis")
        st.markdown(
             """
             This plot shows the cumulative counts from the processed complete dataset.
             It provides a detailed view of the disease progression using data that has been
             cleaned and adjusted for missing/duplicate values.
             """
         )
        fig2 = plot_figures_country_complete(complete_csv_country, start_date, end_date)
        fig2.tight_layout()
        st.pyplot(fig2)

elif page == "Global Insights":
    st.title("Global Insights & Visualizations")

    st.subheader("Total Cases Per Country")
    fig = px.choropleth(
        worldometer_df, locations="Country.Region", locationmode="country names",
        color="TotalCases", hover_name="Country.Region", scope="world",
        title="Total Cases Across the World", color_continuous_scale="Reds"
    )
    st.plotly_chart(fig)

    st.subheader("Continental Death Rates")
    fig_death_rate = Estimated_Death_Rate_by_Continent()
    st.pyplot(fig_death_rate)

elif page == "Counties Analysis":
    st.title("US Counties Analysis")

    # County selection slider
    selected_county = st.sidebar.selectbox("Select a US County", usa_county_wise_df['Admin2'].unique())

    # Split the page into two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 US Counties by Total Deaths and Confirmed Cases")
        fig_top5 = Top_5_US_Counties()
        st.pyplot(fig_top5)

    with col2:
        st.subheader(f"Figures for {selected_county}")
        fig_county = plot_figures_counties_complete(selected_county)
        st.pyplot(fig_county)
