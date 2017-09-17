

# xgb sweeper function

#' despite the fact that this script is within this repo, I want this to be extremely stand-alone.


library(xgboost)
library(dplyr)
library(gtools)
library(ModelMetrics)


# mix up the data
myiris <- iris[sample(1:nrow(iris), nrow(iris)), ]
myiris$id <- paste0("id", sprintf("%03.0f", 1:nrow(myiris)))


# create train
set.seed(1)
train_indx <- sample(1:nrow(myiris), ceiling(0.7 * nrow(myiris)))


#' Specs:
#'   inputs:
#'     - train data (as DMatrix with label already attached)
#'     - grid of params to try
#'     - nfold (default of 5)
#'     - nrounds (default of 10000)
#'     - early_stop_rounds (default of 100)
#'     - (optional) file path to write to disk
#'         - if this is supplied, write results to disk AS they happen (cat)
#'             - beginning of function, overwrite any existing file with this name, then append to it every loop
#'         - if this isn't supplied, return results as a dataframe
#'     - (optional) test data (as DMatrix with label attached (no reason to include test if no label attached))



data_train <- myiris[train_indx, ]
y_train <- data_train[, c("id", "Species")]
y_train$target <- as.integer(as.factor(y_train$Species)) - 1
excl_cols_train <- c("id", "Species")
dmat_train <- xgboost::xgb.DMatrix(as.matrix(data_train[, setdiff(names(data_train), excl_cols_train)]), label=y_train$target)

data_test <- myiris[-train_indx, ]
y_test <- data_test[, c("id", "Species")]
y_test$target <- as.integer(as.factor(y_test$Species)) - 1
excl_cols_test <- c("id", "Species")
dmat_test <- xgboost::xgb.DMatrix(as.matrix(data_test[, setdiff(names(data_train), excl_cols_test)]), label=y_test$target)


dim(dmat_train); dim(dmat_test)



# xgboost parameters for CV and xgbtrain
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 3,
              "eta" = c(0.1),              # learning rate, how far to step in gradient descent movements
              "max_depth" = c(3, 7),
              "subsample" = c(0.7, 0.8),
              "colsample_bytree" = c(0.7),
              "lambda" = c(0, 1.0),
              "alpha" = c(0, 1),             # L1 regularization
              # "min_child_weight" = 6,  
              # "gamma" = 10,            # complexity control (minimum impact on loss function for split to occur)
              "nthread" = 4)             # CPUs to use

xgb_grid <- expand.grid(param)




# FUNCTION ----------------------------------------------------

tjv_xgb_classi_sweep <- function(param_train, param_grid, param_resultpath=NULL, param_nrounds=10000,
                                 param_folds=5, param_early_stop=100) {
    
    
    # # for rapid dev
    # param_train <- dmat_train
    # param_grid <- xgb_grid
    # param_resultpath <- "xgb_sweep_results.csv"  # <-- keeping this simple for now
    # param_nrounds = 10000
    # param_folds = 5
    # param_early_stop=100
    
    
    # if a file path was supplied, create or overwrite the file, otherwise, start a data.frame
    if(!is.null(param_resultpath)) {
        cat("parameterkey, cv_score, best_nrounds\n", file=param_resultpath, append = FALSE)
    } 
    
    
    # iterate for every row in the parameter grid
    for(i in 1:nrow(param_grid)) {
        
        # parameter key
        this_parakey <- paste0(names(param_grid[i, ]), "_", as.list(param_grid[i, ]), collapse="|")
        
        # execute cv
        this_cv <- xgb.cv(
            data = param_train,
            nrounds = param_nrounds,
            nfold = param_folds,
            early_stopping_rounds = param_early_stop,
            params = as.list(param_grid[i,])
        )
        
        
        # capture nrounds and score for min error
        ev_log <- data.frame(this_cv$evaluation_log)
        ev_log_metric <- names(ev_log)[grepl("^test_", names(ev_log)) & grepl("_mean", names(ev_log))]
        this_best_nrounds <- which.min(ev_log[, ev_log_metric])
        this_best_score <- ev_log[this_best_nrounds, ev_log_metric]
        
        
        # if path isn't null, cat to it, if it is, then either create the df (i is 1) or concat to the df (i > 1)
        if(!is.null(param_resultpath)) {
            cat(paste0(this_parakey, ", ", this_best_score, ", ", this_best_nrounds, "\n"), 
                file = param_resultpath, append = TRUE)
        } else {
            if(i == 1) {
                results_df <- data.frame("parameterkey"=this_parakey, "cv_score"=this_best_score, "best_nrounds"=this_best_nrounds)
            } else {
                this_df <- data.frame("parameterkey"=this_parakey, "cv_score"=this_best_score, "best_nrounds"=this_best_nrounds)
                results_df <- bind_rows(results_df, this_df)
            }
        }
        
        
    } # end for loop
    
    
    # if result path is null then return it as a data.frame
    if(is.null(param_resultpath)) {
        return(results_df)
    }
    
    
} # end the function


# FuNCTION TESTING ---------------------------------------------


# write to file path
tjv_xgb_classi_sweep(param_train=dmat_train, 
                     param_grid=xgb_grid, 
                     param_resultpath="xgb_sweep_results.csv", 
                     param_nrounds=10000, 
                     param_folds=5, 
                     param_early_stop=100)


# just return the df
tjv_xgb_classi_sweep(param_train=dmat_train,
                     param_grid=xgb_grid,
                     # param_resultpath = NULL,
                     param_nrounds=10000,
                     param_folds=5,
                     param_early_stop=100
                     )


# PROCEDURAL ---------------------------------------------------
    
    
    param_grid <- expand.grid(param, stringsAsFactors = F)
    
    # two options here, concat the whole param list together as a single string, or place them in separate fields
    strsplit(paste0(names(param_grid[1, ]), "_", as.list(param_grid[1, ]), collapse="|"), "\\|")
    paramkey <- paste0(names(param_grid[1, ]), "_", as.list(param_grid[1, ]), collapse="|")
    
    
    # execute cv
    this_cv <- xgb.cv(
        data = dmat_train,
        nrounds = 10000,
        nfold = 5,
        early_stopping_rounds = 100,
        params = as.list(param_grid[1,])
    )
    
    
    # capture nrounds and score for min error
    ev_log <- data.frame(this_cv$evaluation_log)
    ev_log_metric <- names(ev_log)[grepl("^test_", names(ev_log)) & grepl("_mean", names(ev_log))]
    this_best_nrounds <- which.min(ev_log[, ev_log_metric])
    this_best_score <- ev_log[this_best_nrounds, ev_log_metric]
    
    
    # write it out or return as DF.
    
    
    
    # ONLY IF TEST WAS PROVIDED -- No, this is dumb and over complicated for what I need. Just do a cv which is more informative anyway.
    
        # check if label is included?
        if(!is.null(getinfo(dmat_test, "label"))) {
            print("labels in test data are not null, let's test it")
            
            this_model <- xgboost(
                data = dmat_train,
                nrounds = this_best_nrounds,
                params = as.list(param_grid[1, ])
            )
            
        }
        
        # 
        if(param_grid[1, ]$eval_metric == "mlogloss") {
            y_test_actual <- getinfo(dmat_test, "label")
            yhat_test <- predict(this_model, dmat_test)
            yhat_test_mat <- matrix(yhat_test, ncol=param_grid[1, ]$num_class, byrow=T)
            yhat_test_df <- data.frame(yhat_test_mat)
            mlogLoss(y_test_actual, yhat_test_df)
            
            
        }
        
    
    
    
    
    
    
