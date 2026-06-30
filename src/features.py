"""
Feature Engineering Module
Author: Nadia Ayar Rojas
Description: Handles chronological train-test splitting, feature standardization 
             (Z-score scaling), and dimensionality reduction via RFE.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def split_chronologically(df_aligned: pd.DataFrame, predictor_cols: list, train_end_year: int = 2017, test_start_year: int = 2018) -> tuple:
    """
    Splits the master dataset into chronological Training (<=2017) and 
    Testing (>=2018) sets to protect against forward data leakage.
    """
    train_df = df_aligned[df_aligned['Year'] <= train_end_year].copy()
    test_df = df_aligned[df_aligned['Year'] >= test_start_year].copy()
    
    X_train = train_df[predictor_cols]
    y_train = train_df['Target_Growth']
    X_test = test_df[predictor_cols]
    y_test = test_df['Target_Growth']
    
    # Validation check for data leakage
    train_years = set(train_df['Year'].unique())
    test_years = set(test_df['Year'].unique())
    if not train_years.isdisjoint(test_years):
        print("[WARNING] Temporal overlap detected between Train and Test sets!")
        
    return X_train, y_train, X_test, y_test

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Applies Z-score standardization. Fits the scaler strictly on training 
    parameters and transforms both sets to prevent data leakage.
    """
    scaler = StandardScaler()
    
    # Fit strictly on train, transform both
    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_test_scaled_arr = scaler.transform(X_test)
    
    # Reconstruct DataFrames to keep column metadata intact
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, index=X_test.index, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled

def select_features_rfe(X_train_scaled: pd.DataFrame, y_train: pd.Series, n_features: int = 20) -> list:
    """
    Executes Recursive Feature Elimination (RFE) using a linear estimator 
    to isolate and return the Top N most impactful structural predictors.
    """
    estimator = LinearRegression()
    rfe_selector = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
    
    rfe_selector.fit(X_train_scaled, y_train)
    selected_mask = rfe_selector.support_
    selected_cols = X_train_scaled.columns[selected_mask].tolist()
    
    print(f"[INFO] RFE successfully isolated the Top {len(selected_cols)} predictors.")
    return selected_cols
