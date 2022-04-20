# Machine Learning Applications to Land and Structure Valuation

This repository contains the complete R code of Case Study 1 of the following open-access publication:

Mayer, M.; Bourassa, S.C.; Hoesli, M.; Scognamiglio, D. Machine Learning Applications to Land and Structure Valuation. J. Risk Financial Manag. 2022, 15, 193. https://doi.org/10.3390/jrfm15050193

In the case study, we use a fantastic dataset on 13,000 houses sold in Miami to show that fitting (generalized) structured additive regression models via tree boosting leads to models with excellent interpretability/accuracy trade-off, see above publication for much more information.

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

In case you don't have Python installed, run the following code in R:
```
library(keras)
install_keras()
```

Note: The results of the deep neural net might slightly differ due to seeding of TensorFlow's random generator.

## Errata of above publication

- In subsection "2.1. STAR Models and Deep Learning", line 4, there is a superfluous "which is": The optional inverse link acts on the model output, not on the response Y.
