# Data Provenance & Access Guide

This directory documents the source and process of accessing the raw datasets utilized in this research project.

## Dataset Source

The primary data is derived from the World Bank's **World Development Indicators (WDI)**, a definitive collection of international metrics on development and economic growth. 

To ensure reproducibility, the exact snapshot of the data used in this pipeline is hosted on Kaggle:

* **Dataset Link:** [World Development Indicators - Nov 2025](https://www.kaggle.com/datasets/nadyaayar/world-development-indicators-nov-2025)

## Required Source Files

The computational pipeline expects two specific files from the dataset to be present in the working environment:

1. `WDICSV.csv`: Contains the raw time-series for the indicators across all nations and years.
2. `WDICountry.csv`: Contains country-level metadata for geographical region and income group stratification.
