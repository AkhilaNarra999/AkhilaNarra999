# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:06:50 2024

@author: NARRA
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataframes():
    """
    Load the required dataframes.

    Returns:
    asia_data (pd.DataFrame): Time series data for Asian countries.
    countries_pop (pd.DataFrame): Population data for Asian countries.
    """
    asia_data = pd.read_csv("D:\\New folder\\timeSeries.csv")
    countries_pop = pd.read_csv("D:\\New folder\\population data.csv")

    return asia_data, countries_pop


def preprocess_data(countries_pop, asia_data):
    """
    Preprocess data by removing unnecessary columns and selecting relevant countries.

    Args:
    countries_pop (pd.DataFrame): Population data for Asian countries.
    asia_data (pd.DataFrame): Time series data for Asian countries.

    Returns:
    asia_pop (pd.DataFrame): Processed population data for selected countries.
    cpi (pd.DataFrame): Processed Consumer Price Index data for selected countries.
    debt (pd.DataFrame): Processed debt data for selected countries.
    """
    # Remove unnecessary columns
    countries_pop = countries_pop.drop(
        columns=[
            'Country Code',
            'Indicator Name',
            'Indicator Code'])

    # Select relevant countries
    countries = [country for country in countries_pop['Country Name'].to_numpy(
    ) if country in asia_data['Country Name'].to_numpy()]
    countries = [
        country for country in countries if country not in [
            'Mongolia',
            'Palau',
            'Tonga',
            'South Asia']]

    # Select indicators
    cpi_indicator = 'Prices, Consumer Price Index, All items, CPI (p.a.), percent change'
    debt_indicator = 'Fiscal, General Government, Gross debt position, 2001 Manual, Percent of FY GDP'

    # Filter data for selected countries and indicators
    asia_pop = countries_pop.set_index('Country Name').loc[countries, :]
    cpi = asia_data.loc[asia_data['Indicator Name'] == cpi_indicator, :].drop(
        columns=['Indicator Name', 'Country Code', 'Indicator Code', 'Attribute'])
    cpi = cpi.set_index('Country Name').loc[countries, :]
    debt = asia_data.loc[asia_data['Indicator Name'] == debt_indicator, :].drop(
        columns=['Indicator Name', 'Country Code', 'Indicator Code', 'Attribute'])
    debt = debt.set_index('Country Name').loc[countries, :]

    return asia_pop, cpi, debt


def compute_stats_per_country(dframe):
    """
    Compute summary statistics for each country.

    Args:
    dframe (pd.DataFrame): Input dataframe.

    Returns:
    stats (pd.DataFrame): Summary statistics.
    """
    dframe = dframe.T  # Transpose
    stats = dframe.describe()
    return stats.T


def main():
    # Load data
    asia_data, countries_pop = load_dataframes()

    # Preprocess data
    asia_pop, cpi, debt = preprocess_data(countries_pop, asia_data)

    # Display summary statistics
    print("Summary of Asian Nations CPI")
    print(compute_stats_per_country(cpi))
    print("\nSummary of Asian Nations debt position")
    print(compute_stats_per_country(debt))
    print("\nSummary of Asian Population Statistics")
    print(compute_stats_per_country(asia_pop))

    # Plotting
    sns.set_style("whitegrid")
    figure, axis = plt.subplots(2, 2, figsize=(30, 25), gridspec_kw={
                                'hspace': 0.09, 'wspace': 0.4})  # Decreased hspace

    # Plot 1: Line Plot of CPI Trends for Asian Countries
    cpi.T.plot(ax=axis[0, 0])
    axis[0, 0].set_facecolor('#F0F0F0')
    axis[0,
         0].set_title('Consumer Price Index Trends in Asian Countries',
                      fontsize=12,
                      fontweight='bold',
                      color='white')
    axis[0, 0].set_xlabel("Year", fontsize=12, color='white')
    axis[0, 0].set_ylabel("Change in CPI", fontsize=12, color='white')
    axis[0, 0].legend(loc='upper right', fontsize=12,
                      facecolor='white', edgecolor='white')

    # Plot 2: Scatter Plot of Mean Population (horizontal)
    meanpop = asia_pop.mean(axis=1)
    colors = sns.color_palette("husl", len(meanpop))
    axis[0, 1].set_facecolor('#F0F0F0')
    axis[0, 1].scatter(x=meanpop.values, y=meanpop.index,
                       color=colors, s=80, alpha=0.7, edgecolors='k')
    axis[0, 1].set_title('Mean Population Over Time',
                         fontweight='bold', fontsize=12, color='white')
    axis[0, 1].set_xlabel('Mean Population', fontsize=12, color='white')
    axis[0, 1].set_ylabel('Country Name', fontsize=12, color='white')
    axis[0, 1].grid(axis='x', linestyle='--', alpha=0.6)

    # Annotate each point with the country name
    for i, txt in enumerate(meanpop.index):
        axis[0, 1].annotate(txt, (meanpop.values[i], i), ha='left',
                            va='center', fontsize=12, color='darkslategrey')

    # Plot 3: Violin Plot for Debt Position with different colors
    sns.violinplot(meanpop, ax=axis[1, 0], palette=[
                   "#3498db", "#2ecc71", "#e74c3c", "#f39c12"], inner="quartile")
    axis[1, 0].set_facecolor('#F0F0F0')  # Light gray background color
    axis[1, 0].set_xlabel('Population', fontsize=10, color='white')
    axis[1, 0].set_ylabel('Frequency', fontsize=10, color='white')
    axis[1, 0].set_title("Population Distribution",
                         fontsize=12, fontweight='bold', color='white')

    # Plot 4: Line Plot for Debt Position and Population Trends
    meandebt = debt.mean(axis=1)
    axis[1, 1].set_facecolor('#F0F0F0')
    axis[1, 1].plot(meandebt.index, meandebt, label="Debt Position",
                    color='blue')
    ax4t = axis[1, 1].twinx()
    ax4t.plot(
        meanpop.index,
        meanpop,
        label="Population",
        color='#d55e00',
        linestyle='dashed',
        marker='o')
    axis[1, 1].legend(loc='upper left', fontsize=8,
                      facecolor='white', edgecolor='white')
    ax4t.legend(
        loc='upper right',
        fontsize=10,
        facecolor='white',
        edgecolor='white')
    axis[1, 1].set_yticks(axis[1, 1].get_yticks())
    axis[1, 1].set_yticklabels(
        axis[1, 1].get_yticklabels(), rotation=0, fontsize=10, color='white')
    axis[1, 1].set_xticks(axis[1, 1].get_xticks())
    axis[1, 1].set_xticklabels(
        axis[1, 1].get_xticklabels(), rotation=90, fontsize=8, color='white')
    axis[1, 1].set_xlabel("Countries", fontsize=6,
                          color='white')  # Set x-axis label
    axis[1, 1].set_ylabel("Debt Position", fontsize=10,
                          color='white')  # Set y-axis label
    axis[1, 1].set_title("Debt Position and Population Trends",
                         fontsize=10, fontweight='bold', color='white')

    # Add text to the plots
    text_name = "Analysis of Population, CPI, and Debt"
    t1 = "THIS PLOT ILLUSTRATES ASIAN COUNTRIES':\n"
    t2 = '1. Consumer Price Index Trends Over Time\n'
    t3 = '''    This line plot displays the Consumer Price Index (CPI) trends over time for various Asian countries. CPI is a crucial economic indicator that reflects changes in the average prices of goods and services consumed by households. \n'''
    t4 = '2. Scatter Plot of Mean Population\n'
    t5 = '''    This scatter plot visualizes the mean population of Asian countries over time. Each point on the plot represents the mean population of a country, offering insights into the overall population dynamics across the region. \n'''
    t6 = '3. Population Distribution - Violin Plot\n'
    t7 = '''    The violin plot provides a detailed view of the distribution of populations among Asian countries. It highlights the range and distribution of population sizes, helping to identify patterns and variations. \n'''
    t8 = '4. Debt Position and Population Trends\n'
    t9 = '''    This dual-axis line plot illustrates the trends in both debt position and population for Asian countries. It allows for the comparison of how changes in debt position correlate with population trends.'''
    whole_text = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9

    props_1 = dict(boxstyle='round', facecolor='#FFA500', alpha=0.5)
    props_2 = dict(boxstyle='rarrow', facecolor='#FFA500', alpha=0.5)

    figure.text(
        0.1,
        0.10,
        whole_text,
        ha='left',
        va='top',
        wrap=True,
        fontsize=14,
        bbox=props_1,
        color='white')
    figure.text(
        0.1,
        0.94,
        text_name,
        ha='left',
        va='top',
        wrap=True,
        fontsize=12,
        bbox=props_2,
        color='white')

    figure.suptitle(
        f'Consumer Price Index and Debt Position Relative to the Population',
        fontsize=20,
        fontweight='bold',
        color='white')

    # Add background design to the page
    # Set background color to dark blue (#001F3F)
    figure.set_facecolor('#001F3F')

    plt.suptitle(f'NAME: AKHILA NARRA  \nSTUDENT ID: 22100008   \n\
     "Exploring Socioeconomic Dynamics: Consumer Price Index, Debt Position, and Population Trends in Asian Countries"',
                 fontsize=20, fontweight='bold', color='white')

    # Set tick colors to white
    for ax in axis.flatten():
        ax.tick_params(axis='x', colors='white', labelsize=10, which='both')
        ax.tick_params(axis='y', colors='white', labelsize=10, which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    plt.savefig('D:\\New folder\\22100008.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
