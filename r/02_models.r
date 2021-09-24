library(tidyverse)        # 1.3.1
library(xgboost)          # 1.4.1.1
library(keras)            # 2.4.0
library(mboost)           # 2.9.5
library(MetricsWeighted)  # 0.5.3
library(flashlight)       # 0.8.0

source(file.path("r", "functions.R"))
load("prep.RData", verbose = TRUE)
x_regional <- c(x_coord, x_vars[grep("dist", x_vars)])

#===============================================
# Data split
#===============================================

set.seed(1234321L)
.in <- sample(nrow(prep), round(0.80 * nrow(prep)), replace = FALSE)
train <- prep[.in, ]
test <- prep[-.in, ]
y_train <- train[[y_var]]
y_test <- test[[y_var]]

#===============================================
# OLS (without Lat/Lon)
#===============================================

form <- reformulate(x_vars, y_var) %>%
  update(. ~ . - LONGITUDE - LATITUDE - month_sold + factor(month_sold))
ols <- lm(form, data = train)
summary(ols)

#===============================================
# Component-wise boosting
#===============================================

mboost_form <-
  log_price ~ bols(log_living) + bols(log_land) + bols(log_special) +
    bols(age) + bols(structure_quality) +
    bbs(month_sold) + btree(s_dist_highway, s_dist_rail, dist_ocean,
    log_dist_water, s_dist_center, s_dist_sub, LATITUDE, LONGITUDE,
    tree_controls = partykit::ctree_control(maxdepth = 5))

fit_mboost <- gamboost(
  mboost_form,
  control = boost_control(trace = TRUE, nu = 0.1, mstop=1022),
  data = train
)

# Centering the encoder
regio_mboost_mean <- mean(get_regio_mboost(train))

# Optimizing number of boosting rounds
# cv5f <- cv(model.weights(fit_mboost), type = "kfold", B = 5)
# cvm <- cvrisk(fit_mboost, folds = cv5f)
# mstop(cvm)


#===============================================
# XGBoost (unconstrained)
#===============================================

# Data interface
dtrain <- xgb.DMatrix(
  data.matrix(train[, x_vars]),
  label = y_train
)

# Settings
tune <- FALSE
file_grid <- "grid_unconstrained.RData"

if (tune) {
  # Step 1: find learning rate
  xgb_cv(list(learning_rate = 0.1), verbose = 2)

  # Step 2: Grid search CV
  paramGrid <- make_grid()
  for (i in seq_len(nrow(paramGrid))) { # i = 1
    print(i)
    cvm <- xgb_cv(as.list(paramGrid[i, -(1:2)]))
    paramGrid[i, 1] <- bi <- cvm$best_iteration
    paramGrid[i, 2] <- as.numeric(cvm$evaluation_log[bi, "test_rmse_mean"])
    save(paramGrid, file = file_grid)
  }
}
load(file_grid, verbose = TRUE)

# Step 3: Fit on best params
head(paramGrid <- paramGrid[order(paramGrid$score), ])
params <- paramGrid[1, ]
cat("Best rmse (CV):", params$score) # 0.142
xgb_unconstrained <- xgb_train(
  as.list(params[, -(1:2)]),
  nrounds = params$iteration
)

#===============================================
# XGBoost (constrained)
#===============================================

# Build interaction constraint vector
ic <- c(
  list(which(x_vars %in% x_regional) - 1),
  as.list(which(!(x_vars %in% x_regional)) - 1)
)
# ic <- c(list(x_regional), as.list(setdiff(x_vars, x_regional)))

# Settings
tune <- FALSE
file_grid <- "grid_constrained.RData"

if (tune) {
  # Step 1: find learning rate
  xgb_cv(
    list(learning_rate = 0.1,
         interaction_constraints = ic),
    verbose = 2
  )

  # Step 2: Grid search CV
  paramGrid <- make_grid()
  for (i in seq_len(nrow(paramGrid))) { # i = 1
    print(i)
    params <- as.list(paramGrid[i, -(1:2)])
    params$interaction_constraints <- ic
    cvm <- xgb_cv(params)
    paramGrid[i, 1] <- bi <- cvm$best_iteration
    paramGrid[i, 2] <- as.numeric(cvm$evaluation_log[bi, "test_rmse_mean"])
    save(paramGrid, file = file_grid)
  }
}
load(file_grid, verbose = TRUE)

# Step 3: Fit on best params
head(paramGrid <- paramGrid[order(paramGrid$score), ])
params <- as.list(paramGrid[1, -(1:2)])
params$interaction_constraints <- ic
cat("Best rmse (CV):", paramGrid[1, "score"]) # 0.145
xgb_constrained <- xgb_train(
  params,
  nrounds = paramGrid[1, "iteration"]
)

# Centering the encoder
regio_xgb_mean <- mean(get_regio_xgb(train))

#===============================================
# OLS with XGB loc
#===============================================

train <- train %>%
  mutate(regio_xgb = get_regio_xgb(., regio_xgb_mean))
test <- test %>%
  mutate(regio_xgb = get_regio_xgb(., regio_xgb_mean))

ols_xgb <- lm(log_price ~ log_living + log_land + log_special +
                   age + structure_quality + factor(month_sold) +
                   regio_xgb,
                 data = train)
summary(ols_xgb)

#===============================================
# Neural net with regional encoding
#===============================================

x_dense <- setdiff(x_vars, "month_sold")
x_regional
x_object <- setdiff(x_dense, x_regional)

# Standardize X using X_train
sc <- list(
  center = attr(scale(data.matrix(train[, x_dense])), "scaled:center"),
  scale = attr(scale(data.matrix(train[, x_dense])), "scaled:scale")
)

# Callbacks
cb <- list(
  callback_early_stopping(patience = 20),
  callback_reduce_lr_on_plateau(patience = 5)
)

nn <- new_neural_net(0.003)
summary(nn)

# Fit model
history <- nn %>% fit(
  x = prep_nn(train),
  y = y_train,
  epochs = 105, # 200,
  batch_size = 128,
  #validation_split = 0.2,
  #callbacks = cb
)

# Last layer weights
nn$weights[[length(nn$weights) - 1]]

# Centering the encoder
regio_nn_mean <- mean(get_regio_nn(train))


#===============================================
# Interpretation
#===============================================

if (FALSE) {
  library(SHAPforxgboost)

  # Step 1: select some observations
  X <- data.matrix(prep[sample(nrow(prep), 1000), x_vars])

  # Step 2: Crunch SHAP values
  shap <- shap.prep(xgb_constrained, X_train = X)

  # Step 3: SHAP importance
  shap.plot.summary(shap)

  # Step 4: Loop over dependence plots in decreasing importance
  for (v in shap.importance(shap, names_only = TRUE)) {
    p <- shap.plot.dependence(shap, v, color_feature = "auto",
                              alpha = 0.5, jitter_width = 0.1) +
      ggtitle(v)
    print(p)
  }
}

# Set up explainers
pred_xgb <- function(m, X) predict(m, data.matrix(X[, x_vars]))

fl_ols <- flashlight(
  model = ols,
  label = "OLS",
)

fl_ols_xgb <- flashlight(
  model = ols_xgb,
  label = "OLS with XGB loc",
  predict_function = function(m, X) X %>%
    mutate(regio_xgb = get_regio_xgb(., regio_xgb_mean)) %>%
    predict(m, .)
)

fl_xgb <- flashlight(
  model = xgb_unconstrained,
  label = "XGB (unconstrained)",
  predict_function = pred_xgb
)

fl_xgb_c <- flashlight(
  model = xgb_constrained,
  label = "XGB STAR",
  predict_function = pred_xgb
)

fl_nn <- flashlight(
  model = nn,
  label = "NN STAR",
  predict_function = function(m, X)
    as.numeric(predict(m, prep_nn(X), batch_size = 1000))
)

fl_mboost <- flashlight(
  model = fit_mboost,
  label = "mboost STAR"
)

fls <- multiflashlight(
  list(fl_ols, fl_xgb, fl_xgb_c, fl_ols_xgb, fl_nn, fl_mboost),
  data = test,
  y = "log_price"
)

# Performance
perf <- light_performance(
  fls,
  metrics = list(RMSE = rmse, `R squared` = r_squared),
  reference_mean = mean(y_train) # to get clean test r-squared
)
perf_wide <- perf$data %>%
  pivot_wider(id_cols = "label", names_from = "metric",
              values_from = "value") %>%
  rename(Model = "label")
perf_wide

# Permutation importance
imp <- light_importance(fls, v = x_vars)
plot(imp, fill = "#03336B", facet_scales = "free_x")

# Interaction strength (takes long)
inter <- light_interaction(fls, v = most_important(imp, 4),
                           pairwise = TRUE)
plot(inter, fill = "#03336B")

# ICE curves
light_ice(fls, v = "log_living", n_max = 200, seed = 245) %>%
  plot(alpha = 0.2, color = "#E69F00") +
  ylab("log appraisal")

light_ice(fls, v = "dist_ocean", n_max = 200, seed = 245) %>%
  plot(alpha = 0.2, color = "#E69F00")

# Map of regio effect (full data)
X_nn <- prep %>%
  mutate(regio = get_regio_nn(., regio_nn_mean),
         type = "Neural net")
X_xgb <- prep %>%
  mutate(regio = get_regio_xgb(., regio_xgb_mean),
         type = "XGBoost")
X_mboost <- prep %>%
  mutate(regio = get_regio_mboost(., regio_mboost_mean),
         type = "mboost")
rbind(X_nn, X_xgb, X_mboost) %>%
  mutate(type = factor(type, levels = c("XGBoost", "Neural net", "mboost"))) %>%
ggplot(aes(LONGITUDE, LATITUDE, z = regio)) +
  coord_quickmap() +
  stat_summary_2d(bins = 50) +
  facet_wrap(~ type) +
  labs(fill = "Relative effect",
       x = element_blank(),
       y = element_blank()) +
  scale_fill_viridis_c(option = "inferno", begin = 0.2) +
  theme_void(base_size = 12) +
  theme(axis.ticks = element_blank(),
        axis.text = element_blank(),
        strip.text.x = element_text(size = 14),
        legend.position = c(0.28, 0.2),
        legend.key.height = unit(5, units = "mm"),
        legend.background = element_rect(fill = "transparent",
                                         linetype = "blank"),
        plot.margin=grid::unit(c(0, 0, 0, 0), "mm"))
