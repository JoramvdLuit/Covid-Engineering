# COVID-19 Infections Dashboard
Data Engineering Covid Assignment

By Boris, Marthe Welkzijn & Joram van der Luit
This repository provides a comprehensive analysis and interactive dashboard to study the spread of COVID-19. The project covers data visualization, parameter estimation using an extended SIR model (including a deceased state), data wrangling, and an interactive dashboard built with Streamlit.

Table of Contents
•	Project Overview
•	Project Parts and User Interaction 
  o	Part 1: Data Visualization & SIR Model Simulation
  o	Part 3: Database Interaction & Country Analysis
  o	Part 4: Data Wrangling
  o	Part 5: Interactive Dashboard
  o	Part 6: Version Control and Deployment
•	Prerequisites
•	How to Run the Program
•	Project Structure
•	SIR Model fit test and conclusions

Project Parts and User Interaction
Part 1: Data Visualization & SIR Model Simulation
•	What It Does:
o	Loads time-series data from day_wise.csv and generates line graphs for new cases, deaths, and recoveries over time.
o	Implements a SIR model (with an added Deceased state) and simulates the epidemic using three different parameter sets.
o	Provides functions to filter the time range for visualization.
•	User Interaction:
o	Users can call functions like plot_figure_dates(df, start_date, end_date) to visualize the data within a specific date range.
o	The SIR model simulation functions enable users to plot Covid data by simulating the number of deaths, infected confirmed etc using parameters sets.

Part 4: Data Wrangling
•⁠  ⁠What It Does:
o	Processes an additional dataset (complete.csv) that contains detailed daily counts per country.
o	Addresses missing values by applying manual interpolation and invariant-based corrections.
o	Groups and aggregates data to ensure that cumulative counts are accurate.
•⁠  ⁠User Interaction:
o	Users can choose a country to perform operations on
o	The function plot_figures_country_complete(country, start_date, end_date) now provides plots of cumulative totals (Confirmed, Active, Deaths, and Recovered) based on the cleaned data.
o	Additional functions support detailed analysis at the county level (especially for the USA) via plot_figures_counties_complete(county).
