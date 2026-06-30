# Project Documentation

This directory contains the official academic dissertation submitted to obtain the Master's degree in Artificial Intelligence and Data Science.

## Dissertation Abstract

**Title:** Determinants and Prediction of Global Economic Growth: An Ensemble Model Approach  
**Author:** Nadia Ayar Rojas  

Forecasting economic growth with high precision remains a fundamental challenge for governments, central banks, and the private sector, given the non-linear and dynamic nature of macroeconomic indicators. This dissertation addresses this complexity by developing a robust data science framework capable of navigating the complex dynamics inherent in macroeconomic data. Using a balanced panel of 60 countries stratified by income level from the World Bank’s World Development Indicators (2000–2021), this study implements a rigorous pipeline integrating **hybrid imputation** and Recursive Feature Elimination (**RFE**) to isolate 20 structural predictors. 

### Key Methodology & Findings:
* **Architectures Evaluated:**
  * Ridge Regression (Baseline)
  * Support Vector Regression (SVR)
  * Random Forest (RF)
  * Long Short-Term Memory (LSTM)
  * Hybrid Ensemble (LSTM+RF)
    
* **Top Performer:** The novel **Hybrid Ensemble (LSTM+RF)** achieved superior predictive accuracy with a Root Mean Squared Error (**RMSE**) of **0.0452**, outperforming independent architectures.
* **Explainability:** The application of SHapley Additive exPlanations (**SHAP**) revealed critical structural heterogeneity, identifying that drivers such as fertility and infrastructure access exhibit polarity reversals depending on a nation's development stage.

These findings validate the efficacy of hybrid machine learning architectures for economic forecasting and provide a transparent framework for decoding context-dependent determinants of growth, advocating for stage-specific economic policies rather than universal approaches.

## Files
* `global_economic_growth_dissertation.pdf`: Full-text written dissertation.
