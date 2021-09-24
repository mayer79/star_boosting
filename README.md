# Structured Additive Regression and Tree Boosting

Complete R code of Case Study 1 of the corresponding paper (preprint: http://dx.doi.org/10.2139/ssrn.3924412).

## Content 

The folder "r" contains the following scripts:

- 01_describe.r: Loads and describes data. Stores prepared data.
- 02_models.r: Loads prepared data and runs all models, including explainability.
- function.r: A couple of helper functions.

Furthermore we provide two RData files with the results of the XGBoost tune grids.

Cloning the project should provide runnable code, at least if you have R and Python installed.

## Requirements

- R >= 4
- Python version (for deep neural net): 3.6+
