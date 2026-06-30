"""
Data Ingestion Module
Author: Nadia Ayar Rojas
Description: Handles loading raw CSV files, wide-to-long transformation (melting),
             metadata enrichment, and historical data availability visualization.
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

# Suppress warnings for clean execution output
warnings.filterwarnings('ignore')

def load_raw_data(data_path: str, country_path: str) -> tuple:
    """
    Loads the raw WDI data and country metadata from CSV files.
    """
    if not os.path.exists(data_path) or not os.path.exists(country_path):
        raise FileNotFoundError("Source CSV files missing. Please check your data directory.")
        
    df_raw = pd.read_csv(data_path)
    df_country = pd.read_csv(country_path)
    return df_raw, df_country

def transform_wide_to_long(df_raw: pd.DataFrame, df_country: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms wide-format data into a tidy long-format structure
    and merges country metadata while filtering global aggregates.
    """
    # Identify year columns
    year_cols = [col for col in df_raw.columns if col.isdigit()]
    
    # Melt wide to long
    df_melted = df_raw.melt(
        id_vars=['Country Code', 'Indicator Code'], 
        value_vars=year_cols, 
        var_name='Year', 
        value_name='Value'
    )
    df_melted['Year'] = df_melted['Year'].astype(int)
    
    # Merge metadata and filter out aggregates
    country_meta = df_country[['Country Code', 'Table Name', 'Income Group', 'Region']].drop_duplicates()
    df_enriched = df_melted.merge(country_meta, on='Country Code', how='left')
    df_enriched = df_enriched.dropna(subset=['Region', 'Income Group'])
    
    # Clean column names
    df_enriched.rename(columns={'Table Name': 'Country Name'}, inplace=True)
    return df_enriched

def generate_availability_plot(df_enriched: pd.DataFrame, all_x_codes: list, output_path: str = 'period_justification.png'):
    """
    Generates and saves a bar chart verifying historical data availability 
    to empirically justify the 2000-2021 temporal scope.
    """
    # Filter for predictor codes from 1980 onwards
    plot_data = df_enriched[
        (df_enriched['Indicator Code'].isin(all_x_codes)) & 
        (df_enriched['Year'] >= 1980)
    ].copy()
    
    # Count valid records per year
    availability_counts = plot_data.dropna(subset=['Value']).groupby('Year')['Value'].count()
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(availability_counts.index, availability_counts.values, color='skyblue', edgecolor='black', alpha=0.9)
    plt.title('Historical Data Availability: Total Non-Null Data Points for Predictors', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Valid Data Points', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Scope markers
    plt.axvline(x=2000, color='blue', linestyle='--', linewidth=1.5)
    plt.axvline(x=2021, color='blue', linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[INFO] Availability plot exported successfully to {output_path}")
