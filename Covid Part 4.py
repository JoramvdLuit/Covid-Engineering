import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
import statsmodels.api as sm
import plotly.express as px


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

    # removes duplicate values
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
    return df

def plot_figures_country_complete(country):
    df = process_country_complete(country).copy()
    
    df['Confirmed_change'] = df['Confirmed'].diff()
    df['Active_change'] = df['Active'].diff()
    df['Deaths_change'] = df['Deaths'].diff()
    df['Recovered_change'] = df['Recovered'].diff()
    
    plt.figure(figsize=(10, 16))
    
    plt.subplot(4, 1, 1)
    plt.plot(df['Date'], df['Confirmed_change'], label="Daily Change in Confirmed", color='purple')
    plt.xlabel("Date")
    plt.ylabel("Confirmed Change")
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(df['Date'], df['Active_change'], label="Daily Change in Active", color='blue')
    plt.xlabel("Date")
    plt.ylabel("Active Change")
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(df['Date'], df['Deaths_change'], label="Daily Change in Deaths", color='red')
    plt.xlabel("Date")
    plt.ylabel("Deaths Change")
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(df['Date'], df['Recovered_change'], label="Daily Change in Recovered", color='green')
    plt.xlabel("Date")
    plt.ylabel("Recovered Change")
    plt.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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
    
    plt.figure(figsize=(10, 16))
    
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
    plt.show()


plot_figures_country_complete("Netherlands")
plot_figures_counties_complete("Hudson")
