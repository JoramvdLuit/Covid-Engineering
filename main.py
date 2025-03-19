import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import statsmodels.api as sm
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide", initial_sidebar_state="expanded")

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                        #Part 1
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

df_daywise = pd.read_csv("day_wise.csv", parse_dates=["Date"])

def plot_figure(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(df["Date"], df["New.cases"], color="blue")
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


def print_sir_model_simulation(df, alpha, beta, gamma, mu, I0, R0, S0, D0, N, parameterset):
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

    print("Estimate for reproduction number R0: ",beta/gamma )
    
    print("Mean Squared Errors for parameterset ", parameterset, ", offset by 1/10000000000000:")
    print("S: ", mse_S / 10000000000000)
    print("I: ", mse_I / 10000000000000)
    print("R: ", mse_R / 10000000000000)
    print("D: ", mse_D / 10000000000000)

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


        S_sim = np.array(S)
        
    fig = plt.figure(figsize=(10, 6))
    plt.plot(time, S, label="Susceptible")
    plt.plot(time, I, label="Infected")
    plt.plot(time, R, label="Recovered")
    plt.plot(time, D, label="Deceased")
    plt.xlabel("Date")
    plt.ylabel("Number of individuals")
    plt.title(f"SIR Model with Deaths Simulation using parameterset: {parameterset}")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    return fig

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                        #Part 3
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
    plt.tight_layout()

    axes[1].plot(df_filtered['Date'], df_filtered['Deaths_fraction'], label=f"Cumulative Deaths / Population, {country}", color='red')
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deaths Fraction (Total)")
    axes[1].legend()
    plt.tight_layout()

    axes[2].plot(df_filtered['Date'], df_filtered['Recovered_fraction'], label=f"Cumulative Recovered / Population, {country}", color='green')
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Recovered Fraction (Total)")
    axes[2].legend()
    plt.tight_layout()
    
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

    # Print the final filled rows.
    print("Processed DataFrame for", country)
    print(df[['Date', 'Confirmed', 'Active', 'Deaths', 'Recovered']])
    return df_complete

def plot_figures_country_complete(country, start_date, end_date):
    df = process_country_complete(country).copy()

    interval = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    df_filtered = df.loc[interval].copy()

    df_filtered['Confirmed_change'] = df_filtered['Confirmed'].diff()
    df_filtered['Active_change'] = df_filtered['Active'].diff()
    df_filtered['Deaths_change'] = df_filtered['Deaths'].diff()
    df_filtered['Recovered_change'] = df_filtered['Recovered'].diff()
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    axes[0].plot(df_filtered["Date"], df_filtered["Confirmed_change"], label="Daily Change in Confirmed", color="purple")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Confirmed Change")
    axes[0].legend()
    plt.tight_layout()

    axes[1].plot(df_filtered["Date"], df_filtered["Active_change"], label="Daily Change in Active", color="blue")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Active Change")
    axes[1].legend()
    plt.tight_layout()

    axes[2].plot(df_filtered["Date"], df_filtered["Deaths_change"], label="Daily Change in Deaths", color="red")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Deaths Change")
    axes[2].legend()
    plt.tight_layout()

    axes[3].plot(df_filtered["Date"], df_filtered["Recovered_change"], label="Daily Change in Recovered", color="green")
    axes[3].set_xlabel("Date")
    axes[3].set_ylabel("Recovered Change")
    axes[3].legend()
    plt.tight_layout()
    
    return fig

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
    mu_t = df['Deaths_change'].iloc[1:] / df['Active'].iloc[1:]
    alpha_t = (gamma * df['Active'].iloc[1:] - df['Recovered_change'].iloc[1:]) / df['Recovered_change'].iloc[1:]
    beta_t = (df['Active_change'].iloc[1:] / df['Active'].iloc[1:] + mu_t + gamma) / S_t.iloc[1:]
    R0 = beta_t / gamma

    params = [alpha_t, beta_t, gamma, mu_t, S_t]


    return params

def plot_figures_counties_complete(county):
    county_df = usa_county_wise_df[usa_county_wise_df['Admin2'] == county].copy()
    
    county_df['Date'] = pd.to_datetime(county_df['Date'])
    county_df.sort_values("Date", inplace=True)
    
    fig = plt.figure(figsize=(10, 16))
    
    plt.subplot(2, 1, 1)
    plt.plot(county_df['Date'], county_df['Confirmed'], label="Daily Change in Confirmed", color='purple')
    plt.xlabel("Date")
    plt.ylabel("Confirmed Change")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(county_df['Date'], county_df['Deaths'], label="Daily Change in Deaths", color='blue')
    plt.xlabel("Date")
    plt.ylabel("Deaths Change")
    plt.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


#Part 5
# Streamlit Dashboard Configuration
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "SIR Model", "Country Analysis", "Global Insights"])

start_date, end_date = st.sidebar.date_input("Select Date Range", [df_daywise["Date"].min(), df_daywise["Date"].max()])
selected_country = st.sidebar.selectbox("Select a country", worldometer_df["Country.Region"].unique())

complete_csv_country = "US" if selected_country == "USA" else selected_country

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

elif page == "SIR Model":
    st.title("SIR Model for COVID-19 Spread (Country-Specific)")

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

    We can then use the model to estimate the basic reproduction number R0 given by β(t) / γ
    """)

    # Ensure "USA" maps to "US" for complete.csv
    complete_csv_country = "US" if selected_country == "USA" else selected_country

    # Estimate country-specific parameters
    params = estimate_parameters_by_country(selected_country)
    alpha_t, beta_t, gamma, mu_t, S_t = params

    # Compute R₀ trajectory
    R0_traj_df = R0_trajectory_by_country(selected_country)
    R0_estimate = beta_t / gamma

    # Display parameters
    st.subheader(f"SIR Model for {selected_country}")

    # Plot R0 trajectory
    st.subheader(f"R₀ Trajectory for {selected_country}")
    fig_R0 = plt.figure(figsize=(10, 5))
    plt.plot(R0_traj_df["Date"], R0_traj_df["R0"], label="R₀ Estimate", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Estimated R₀")
    plt.title(f"R₀ Trajectory for {selected_country}")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig_R0)
        
elif page == "Country Analysis":
    st.title(f"COVID-19 Analysis for {selected_country}")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_totals_for_country(selected_country, start_date, end_date)
        st.pyplot(fig)
    with col2:
        fig = plot_figures_country_complete(complete_csv_country, start_date, end_date)
        st.pyplot(fig)

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



def main():
    plot_figure(df_daywise)

    start_date = "2020-03-01"
    end_date = "2020-05-01"
    plot_figure_dates(df_daywise, start_date, end_date)

    parameter_sets = [params1, params2, params3]
    for i, params in enumerate(parameter_sets, start=1):
        graph_sir_model_simulation(df_daywise, params[0], params[1], params[2], params[3],
                                I0, R0, S0, D0, N, i)
        

    #Part3
    start_date = "2020-01-22"
    end_date = "2020-07-27"
    plot_totals_for_country("Netherlands", start_date, end_date)
    plt.show()

    print(estimate_parameters_by_country("Netherlands"))
    R0_traj_df = R0_trajectory_by_country("Netherlands")
    
    Active_Cases_fraction_Europe().show()
    Estimated_Death_Rate_by_Continent()
    plt.show()
    Top_5_US_Counties()
    plt.show()

    
    #Part 4
    plot_figures_country_complete("Netherlands", start_date, end_date)
    plot_figures_counties_complete("Hudson")
    #plt.show()

main()
