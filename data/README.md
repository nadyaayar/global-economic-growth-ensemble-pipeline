# Data Layer: World Development Indicators (WDI)

Due to storage policies and the volume of the primary source files from the World Bank database, the raw datasets used in this research are not hosted directly within this GitHub repository.

### Dataset Access
The panel of historical macroeconomic data is publicly hosted and documented on Kaggle:
* **Direct Link:** [World Development Indicators - Nov 2025](https://www.kaggle.com/datasets/nadyaayar/world-development-indicators-nov-2025)

### Files Required for Local Execution:
To execute the end-to-end automated pipeline ('main.py'), download the following files from the Kaggle link above:
1. 'WDICSV.csv' - Contains the raw time-series for the indicators across all nations and years.
2. 'WDICountry.csv' - Contains country-level metadata for geographical region and income group stratification.
