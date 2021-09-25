# Structured Additive Regression and Tree Boosting

This repository contains the complete R code of Case Study 1 of the following preprint:

"Structured Additive Regression and Tree Boosting" by Mayer, Bourassa, Hoesli, and Scognamiglio (2021), 
http://dx.doi.org/10.2139/ssrn.3924412.

In the case study, we use a fantastic dataset with information on 13,000 houses sold in Miami in 2016 to show that fitting structured additive regression models via tree boosting leads to models with excellent interpretability/accuracy trade-off. Structured additive regression (STAR) is a generalization of the generalized additive model.
The dataset was kindly provided by our coauthor Prof. Steven Bourassa and made publicly available for research purposes on https://www.openml.org/d/43093 .

## Content 

The folder "r" contains the following scripts:

- 01_describe.r: Downloads house price data, prepares and describes it. Stores prepared data.
- 02_models.r: Loads prepared data and runs all models, including explainability.
- function.r: Helper functions.

Two RData files contain the parameter grids of the XGBoost models, so you don't need to tune these models again.

## Requirements

Cloning the repo will provide runnable R code, given you have R and Python installed.

- R >= 4
- Python version (for deep neural net): 3.6+
- R packages: tidyverse, xgboost, mboost, keras, flashlight

In case you don't have Python installed, simply comment out the code in 02_models.r related to the deep neural net.
