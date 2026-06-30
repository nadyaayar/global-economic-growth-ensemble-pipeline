# Research & Experimentation Notebook

This directory contains the core experimental framework and computational pipeline for the dissertation project.

## Research Pipeline Overview

The primary notebook executes an end-to-end Machine Learning and Deep Learning workflow structured into the following phases:

* **Step 0 & 1: Environment Setup & Configuration**
  * Initialization of the technical stack (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `shap`) paired with strict determinism configurations to ensure environment reproducibility.
  * Definition of the 60 candidate World Development Indicators (WDI) mapped across six macroeconomic themes alongside helper dictionaries to translate raw codes into human-readable features.

* **Step 2 & 3: Data Ingestion & Availability Analysis**
  * Ingestion of raw data and structural transformation from wide-format to tidy long-format data using the `melt` function, supplemented by metadata enrichment for regions and income groups.
  * Generation of the historical data availability visualization to empirically support the 2000–2021 temporal scope.
  
* **Step 4 & 5: Dual-Stage Filtering & Feature Engineering**
  * Quality rule: Exclusion of countries with $<75\%$ data availability across the 22-year window.
  * Relevance rule: Selection of the Top 15 largest economies per income group, establishing a balanced panel of 60 nations.
  * Engineering of the target variable (`Target_Growth`) using log-difference transformation.

* **Step 6 & 7: Hybrid Imputation & Temporal Alignment**
  * Application of a two-phase imputation strategy: intra-country Linear Interpolation + k-Nearest Neighbors (KNN) Imputation, verified by an artificial masking simulation.
  * Implementation of a 1-period temporal lag ($t-1$) applied to all 60 explanatory variables to eliminate reverse causality and guarantee a strict predictive structure.

* **Step 8, 9 & 10: Chronological Splitting, Scaling & Dimension Reduction**
  * Train-Test split partitioned by time (Train: 2001–2017; Test: 2018–2021) to eliminate data leakage.
  * Z-score standardization fitted strictly on training parameters.
  * Application of Recursive Feature Elimination (RFE) using a linear estimator to isolate the Top 20 most impactful structural features.

* **Step 11: Comparative Modeling & Hybrid Ensemble**
  * Performance evaluation and hyperparameter tuning (`GridSearchCV` with `TimeSeriesSplit`) across independent architectures: Ridge Regression, Support Vector Regression, Random Forest, and a LSTM network.
  * Evaluation of the winning **Hybrid Ensemble (LSTM + RF)** against standard regression metrics ($RMSE$, $MAE$, $R^2$).

* **Step 12 & 13: Interpretability & Diagnostics**
  * Post-hoc global and stratified feature importance decoding using SHAP explainers to capture structural heterogeneity and driving forces across different country income levels.
  * Dataset Shift analysis using Kernel Density Estimation (KDE) to visually and statistically isolate the 2020 structural break caused by the COVID-19 pandemic.

## Files
* `global_economic_growth_ensemble_pipeline.ipynb`: The complete, self-contained Jupyter Notebook executing the pipeline detailed above.
