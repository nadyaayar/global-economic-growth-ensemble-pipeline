"""
Data Processing Module
Author: Nadia Ayar Rojas
Description: Handles dual-stage country filtering, matrix reshaping, target variable 
             engineering (log differences), hybrid imputation, and temporal lagging.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score

def filter_countries(df_enriched: pd.DataFrame, all_x_codes: list, ranking_code: str, start_year: int = 2000, end_year: int = 2021) -> list:
    """
    Executes the dual-stage country selection algorithm:
    Stage 1: Filters countries with >= 75% historical data availability.
    Stage 2: Selects the Top 15 largest economies per income group based on final year GDP.
    """
    df_period = df_enriched[(df_enriched['Year'] >= start_year) & (df_enriched['Year'] <= end_year)].copy()
    
    # Stage 1: Quality Filter
    x_data = df_period[df_period['Indicator Code'].isin(all_x_codes)]
    data_matrix = x_data.pivot_table(index='Country Code', columns='Indicator Code', values='Value', aggfunc='count').fillna(0)
    
    theoretical_max = len(all_x_codes) * (end_year - start_year + 1)
    data_matrix['availability_pct'] = (data_matrix.sum(axis=1) / theoretical_max) * 100
    candidates_stage_1 = data_matrix[data_matrix['availability_pct'] >= 75.0].index.tolist()
    
    # Stage 2: Economic Relevance Filter
    df_countries_only = df_period.drop_duplicates(subset=['Country Code', 'Income Group', 'Country Name'])
    gdp_last_year = df_period[(df_period['Indicator Code'] == ranking_code) & (df_period['Year'] == end_year)]
    gdp_map = gdp_last_year.set_index('Country Code')['Value'].to_dict()
    
    candidates_df = df_countries_only[df_countries_only['Country Code'].isin(candidates_stage_1)].copy()
    candidates_df['GDP_Final_Year'] = candidates_df['Country Code'].map(gdp_map).fillna(0)
    
    final_country_codes = []
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    
    for group in income_groups:
        group_df = candidates_df[candidates_df['Income Group'] == group]
        top_15 = group_df.nlargest(15, 'GDP_Final_Year')
        final_country_codes.extend(top_15['Country Code'].tolist())
        
    return final_country_codes

def construct_and_engineer_target(df_enriched: pd.DataFrame, selected_countries: list, name_map: dict, target_code: str, ranking_code: str, start_year: int = 2000, end_year: int = 2021) -> pd.DataFrame:
    """
    Pivots data from long to wide, drops ranking features, and calculates 
    the economic growth target using chronological log-differences.
    """
    df_filtered = df_enriched[
        (df_enriched['Country Code'].isin(selected_countries)) & 
        (df_enriched['Year'] >= start_year) & 
        (df_enriched['Year'] <= end_year)
    ].copy()
    
    # Translate codes and pivot
    df_filtered['Variable Name'] = df_filtered['Indicator Code'].map(name_map)
    df_wide = df_filtered.pivot_table(index=['Country Name', 'Country Code', 'Year'], columns='Variable Name', values='Value').reset_index()
    
    # Drop ranking helper
    ranking_name = name_map.get(ranking_code)
    if ranking_name in df_wide.columns:
        df_wide.drop(columns=[ranking_name], inplace=True)
        
    # Target engineering via log differences grouped by country
    target_name = name_map[target_code]
    df_wide.sort_values(by=['Country Code', 'Year'], inplace=True)
    df_wide['Log_GDP'] = np.log(df_wide[target_name])
    df_wide['Target_Growth'] = df_wide.groupby('Country Code')['Log_GDP'].diff()
    
    return df_wide

def run_hybrid_imputation(df_wide: pd.DataFrame, predictor_cols: list, seed: int = 42) -> pd.DataFrame:
    """
    Executes intra-country Linear Interpolation followed by distance-weighted KNN (k=10).
    Includes an internal masking simulation to validate imputation accuracy.
    """
    df_imputed = df_wide.copy()
    
    # Phase 1: Linear Interpolation (Intra-country)
    df_imputed[predictor_cols] = df_imputed.groupby('Country Code')[predictor_cols].transform(
        lambda g: g.interpolate(method='linear', limit_direction='both')
    )
    
    # Phase 2: KNN Imputation (if missing values persist)
    if df_imputed[predictor_cols].isna().sum().sum() > 0:
        # Imputation Quality Test (Masking Simulation)
        df_test = df_imputed[predictor_cols].dropna().copy()
        if not df_test.empty:
            np.random.seed(seed)
            mask = np.random.rand(*df_test.shape) < 0.05
            true_vals = df_test.values[mask]
            df_corrupted = df_test.copy()
            df_corrupted.values[mask] = np.nan
            
            test_imputer = KNNImputer(n_neighbors=10, weights='distance')
            guessed_vals = test_imputer.fit_transform(df_corrupted)[mask]
            
            nrmse = np.sqrt(mean_squared_error(true_vals, guessed_vals)) / np.std(true_vals)
            r2 = r2_score(true_vals, guessed_vals)
            print(f"[INFO] Imputation Quality Test - NRMSE: {nrmse:.4f}, R2: {r2:.4f}")

        # Imputation Layer
        imputer = KNNImputer(n_neighbors=10, weights='distance')
        df_imputed[predictor_cols] = imputer.fit_transform(df_imputed[predictor_cols])
        
    return df_imputed

def apply_temporal_alignment(df_imputed: pd.DataFrame, predictor_cols: list) -> pd.DataFrame:
    """
    Applies a 1-period lag (t-1) to predictor features to safeguard against 
    reverse causality, removing resultant incomplete border years.
    """
    df_aligned = df_imputed.copy()
    df_aligned.sort_values(by=['Country Code', 'Year'], inplace=True)
    
    # Shift predictors chronologically per country
    df_aligned[predictor_cols] = df_aligned.groupby('Country Code')[predictor_cols].shift(1)
    
    # Drop rows without targets or features due to shifting
    df_aligned = df_aligned.dropna(subset=predictor_cols + ['Target_Growth'])
    return df_aligned
