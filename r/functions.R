# Helper functions

# Create a grid for random grid search
make_grid <- function(nmax = 20) {
  paramGrid <- expand.grid(
    iteration = NA,
    score = NA,
    learning_rate = 0.1,
    max_depth = 4:6,
    min_child_weight = c(0, 1e-04),
    colsample_bynode = c(0.8, 1),
    subsample = c(0.8, 1),
    reg_lambda = 0:2,
    reg_alpha = 0:2,
    eval_metric = "rmse"
  )
  if (nrow(paramGrid) > nmax) {
    set.seed(342267)
    paramGrid <- paramGrid[sample(nrow(paramGrid), nmax), ]
  }
  paramGrid
}

# wrapper for xgb.cv
xgb_cv <- function(params, verbose = 0) {
  xgb.cv(
    params,
    dtrain,
    nrounds = 5000,
    nfold = 5,
    objective = "reg:squarederror",
    showsd = FALSE,
    early_stopping_rounds = 20,
    verbose = verbose
  )
}

# Wrapper for xgb.train
xgb_train <- function(params, nrounds) {
  set.seed(93845)
  xgb.train(
    params,
    data = dtrain,
    nrounds = nrounds,
    objective = "reg:squarederror"
  )
}

# Typical object without location info
typical_object <- function() {
  data.frame(
    log_living = log(2000),
    log_land = log(7500),
    log_special = 2000,
    age = 30,
    month_sold = 7,
    structure_quality = 4
  )
}

# Function to calculate regio_xgb
get_regio_xgb <- function(X, center = 0) {
  X_plugin <- typical_object() %>%
    cbind(X[, x_regional])
  predict(xgb_constrained, data.matrix(X_plugin[, x_vars])) - center
}

# Function that maps data.frame to scaled network input
prep_nn <- function(X) {
  X_dense <- data.matrix(X[, x_dense, drop = FALSE])
  X_dense <- scale(X_dense, center = sc$center, scale = sc$scale)
  list(
    obj = X_dense[, x_object],
    regional = X_dense[, x_regional],
    time = as.integer(X[["month_sold"]]) - 1L
  )
}

# Initialize neural net
new_neural_net <- function(lr = 0.001) {
  k_clear_session()
  set.seed(1000)
  if ("set_seed" %in% names(tensorflow::tf$random)) {
    tensorflow::tf$random$set_seed(1000)
  } else if ("set_random_seed" %in% names(tensorflow::tf$random)) {
    tensorflow::tf$random$set_random_seed(1000)
  } else {
    print("Check tf version")
  }

  # Model architecture
  regional_input <- layer_input(8, name = "regional", dtype = "float32")
  object_input <- layer_input(5, name = "obj", dtype = "float32")
  time_input <- layer_input(1, name = "time", dtype = "int8")

  time_emb <- time_input %>%
    layer_embedding(12, 1) %>%
    layer_flatten()

  regional_encoding <- regional_input %>%
    layer_dense(12, activation = "tanh") %>%
    layer_dense(6, activation = "tanh") %>%
    layer_dense(3, activation = "tanh") %>%
    layer_dense(1, activation = "linear",
                name = "regional_encoding")

  outputs <- list(time_emb, regional_encoding, object_input) %>%
    layer_concatenate() %>%
    layer_dense(1, activation = "linear")

  inputs <- list(
    obj = object_input,
    regional = regional_input,
    time = time_input
  )

  model <- keras_model(inputs, outputs)

  model %>%
    compile(loss = loss_mean_squared_error,
            optimizer = optimizer_nadam(lr = lr))

  return(model)
}

# Function to calculate regio_nn
get_regio_nn <- function(X, center = 0) {
  X_plugin <- typical_object() %>%
    cbind(X[, x_regional])

  as.numeric(
    predict(nn,
            prep_nn(X_plugin[, x_vars]),
            batch_size = 100)
  ) - center
}

# Function to calculate regio_mboost
get_regio_mboost <- function(X, center = 0) {
  X_plugin <- typical_object() %>%
    cbind(X[, x_regional])
  predict(fit_mboost, X_plugin) - center
}


