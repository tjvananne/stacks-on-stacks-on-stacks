

# 02_ensemb_stage1_multi_function


# if we need to jump INTO the function interactively, use these
    # # for testing, we want to just have the parameter values available to us interactively
    # p_feats_all <- feats_all
    # p_stack_y <- this_stack_y
    # p_tar_var <- "species"
    # p_train_ids <- df_all$id[df_all$dataset == 'train']
    # # p_test_ids  <- df_all$id[df_all$dataset == 'test']  # <-- test what happens if this is null (we don't always care to include test)
    # # inclusion of p_test_ids is useful for instance if we have NA values that we'd like to "predict" on using other features, you can
    # # think of that like a very advanced "stacking" method for missing value imputation
    # p_test_ids <- NULL
    # p_xgb_params <- params
    # p_cv_folds <- 4
    # p_cv_rounds <- 3000
    # p_cv_earlystop <- 15
    # p_cheaters <- these_cheaters
    # p_read_from_cache = TRUE
    # p_stack_identifier = "02"


# FUNC DEF ---------------------------------------------------------------
    #' Note: the dataframe you pass in is in long form and has the following columns:
    #'   - id: unique identifier for an observation in your data
    #'   - feature_name: the name of a specific feature
    #'   - value: the value that corresponds to the intersection of the id and feature_name in the data
    #'   
    #' You should also have an object called "y" in your environment with the following columns:
    #'   - id: unique identifier for an observation in your data (same as above)
    #'   - <target variable name>: this will obviously vary


layer1_multi <- function(
    p_feats_all,              # <-- no longer using global version, pass the whole df in
    p_stack_y,                # <-- the y's relative to THIS stack, not for the overall project
    p_tar_var,                
    p_train_ids,              
    p_test_ids=NULL,          # user doesn't have to supply test ids, useful if they want to use all of train
    p_xgb_params,
    p_cv_folds=4,
    p_cv_rounds=3000,
    p_cv_earlystop=15,
    p_cheaters,               # <-- c("") -- vector
    p_read_from_cache=FALSE,  
    p_stack_identifier
) {
    
    # branch - no_test_data
    use_test <- !is.null(p_test_ids)
    
    print("Initializing classification stage 1 stacker...")
    
    # make sure "id", "feature_name" and "value" are all inside of the long df passed in
    assertthat::assert_that(!any(!c("id", "feature_name", "value") %in% names(p_feats_all)))
    
    # remove NA values and cheater values
    x_all <- subset(feats_all, (!is.na(value) & (!feature_name %in% p_cheaters)))    
    
    # create feature mapping
    x_all$feature_id <- as.numeric(as.factor(x_all$feature_name))    
    # where is this used?
    x_all_feature <- x_all[!duplicated(x_all$feature_id), c("feature_id", "feature_name")]  %>%
        arrange(feature_id)
    
    # split em up
    x_train <- subset(x_all, id %in% p_train_ids)
    x_train$dataset <- 'train'
    x_train <- subset(x_train, !is.na(value))
    if(use_test) {
        x_test <- subset(x_all, id %in% p_test_ids)  
        x_test$dataset <- 'test'
        x_test <- subset(x_test, !is.na(value))
    }
    
    # feature intersection between groups -- only necessary if test group is also present
    if(use_test) {
        x_feat_intersect <- dplyr::intersect(x_train$feature_name, x_test$feature_name)
        x_train <- subset(x_train, feature_name %in% x_feat_intersect) 
        x_test <- subset(x_test, feature_name %in% x_feat_intersect)
        x_all <- dplyr::bind_rows(x_train, x_test) 
    } else {
        x_all <- x_train
    }
    
    
    # numeric factorized id's generated within each group -- NOT UNIQUE IDENTIFIERS
    x_train$id_num <- as.numeric(as.factor(x_train$id))
    if(use_test) x_test$id_num <- as.numeric(as.factor(x_test$id))
    
    # create id mappings for train and test separately
    x_train_id <- x_train %>% select(id, id_num) %>% unique() %>% arrange(id_num)
    if(use_test) x_test_id <- x_test %>% select(id, id_num) %>% unique() %>% arrange(id_num)
    
    # map the answers to each id set
    x_train_id <- merge(x=x_train_id, y=p_stack_y, by="id", all.x=T, all.y=F)
    if(use_test) x_test_id <- merge(x=x_test_id, y=p_stack_y, by="id", all.x=T, all.y=F)    
    names(x_train_id) <- c("id", "id_num", "this_target")
    if(use_test) names(x_test_id) <- c("id", "id_num", "this_target")
    
    # arrange all of these at the same time before generating sparse matrices
    x_train <- x_train %>% arrange(id_num)
    x_train_id <- x_train_id %>% arrange(id_num)
    if(use_test) x_test <- x_test %>% arrange(id_num)
    if(use_test) x_test_id <- x_test_id %>% arrange(id_num)
    y_train <- x_train_id$this_target
    if(use_test) y_test <- x_test_id$this_target
    
    # levels of the target variable
    y_train_faclev <- levels(as.factor(y_train))  # <-- this really only matters for train if you think about it...
    
    # update this just in case it needs it, should be based on train alone
    p_xgb_params$num_class <- length(unique(y_train))
    
    # generate sparse matrices
    x_train_sp <- sparseMatrix(i = x_train$id_num, j = x_train$feature_id, x = x_train$value)
    if(use_test) x_test_sp <- sparseMatrix(i = x_test$id_num, j = x_test$feature_id, x = x_test$value)
    
    
        # assertions to keep the function from silent errors
        assert_that(nrow(x_train_sp) == nrow(x_train_id) & 
                    nrow(x_train_sp) == length(unique(x_train$id)) &
                    length(y_train) == nrow(x_train_sp))
        
        if(use_test) {
            assert_that(nrow(x_test_sp) == nrow(x_test_id) & 
            nrow(x_test_sp) == length(unique(x_test$id)) &
            length(y_test) == nrow(x_test_sp))
        }
    
    
    # manual cross validation
    gc()
    print("Begin manual cross validation...")
    x_cv <- caret::createFolds(1:nrow(x_train_sp), k = p_cv_folds)
    
    # collectors
    x_stack_train_folds <- data.frame()
    if(use_test) x_stack_test <- data.frame()  # <-- going to pool these and then bag them (average) by ID
    
    # dmat for test can 
    if(use_test) dx_test <- xgb.DMatrix(x_test_sp)
    
    for(i in 1:p_cv_folds) {
        
        fold_indx <- x_cv[[i]]
        
        # create in-fold train data
        x_train_fold_id_ <- x_train_id[fold_indx, ]
        x_train_fold_sp_ <- x_train_sp[fold_indx, ]
        dx_train_fold_ <- xgb.DMatrix(x_train_fold_sp_)
        
        # create out-of-fold train data
        x_train_sp_ <- x_train_sp[-fold_indx, ]
        y_train_ <- as.integer(as.factor(y_train[-fold_indx])) - 1
        dx_train_ <- xgb.DMatrix(x_train_sp_, label = y_train_)
        
            # for debugging:
            # x_train_id[fold_indx, ]
        
        
        # cross validation
        cv_filepath <- paste0("../cache/bst_cv_xgbclassi_", p_stack_identifier, ".rds")
        if(i == 1) {
            
            # read from cache if flag is TRUE and if the file is actually there
            if(p_read_from_cache & file.exists(cv_filepath)) {
                bst_cv_ <- readRDS(cv_filepath)
            } else {
                
                # cv to find optimal number of rounds
                bst_cv_ <- xgb.cv(params=params, data=dx_train_, nfold=p_cv_folds, 
                                  nrounds=p_cv_rounds, early_stopping_rounds = p_cv_earlystop)
                
                # cache it
                saveRDS(bst_cv_, paste0("../cache/bst_cv_xgbclassi_", p_stack_identifier, ".rds"))
                
            }
            
        }
        
        
        # this part heavily depends on xgboost not changing their eval log output in xgb.cv
        eval_log_ <- data.frame(bst_cv_$evaluation_log)
        best_nrounds <-  which.min(eval_log_[, grepl("^test_", names(eval_log_)) & grepl("_mean$", names(eval_log_))])
        
        # build model    
        xgbmod <- xgboost(data=dx_train_, param=params, nround=best_nrounds, save_period=NULL)
        
        # visualize feature importance
        if(i == 1) {
            xgb_imp <- xgb.importance(feature_names = unique(x_train$feature_name), model=xgbmod)
            xgboost::xgb.ggplot.importance(xgb_imp)
        }
        
        # predict on this fold of train 
        ypred_fold <- predict(xgbmod, dx_train_fold_)
        ypred_fold_mat <- matrix(ypred_fold, nrow=nrow(x_train_fold_sp_), ncol=p_xgb_params$num_class, byrow = T)
        ypred_fold_df <- as.data.frame(ypred_fold_mat)
        names(ypred_fold_df) <- paste0(y_train_faclev, "_pred_model_", p_stack_identifier)
            # debug01_ <- cbind(ypred_fold_df, x_train_fold_id_)
        ypred_fold_df <- cbind(ypred_fold_df, id=x_train_fold_id_[, "id"])
        x_stack_train_folds <- bind_rows(x_stack_train_folds, ypred_fold_df)
        
        # predict on test
        if(use_test) {
            ypred_test <- predict(xgbmod, dx_test)    
            ypred_test_mat <- matrix(ypred_test, nrow=nrow(x_test_sp), ncol=p_xgb_params$num_class, byrow=T)   
            ypred_test_df <- as.data.frame(ypred_test_mat)
            names(ypred_test_df) <- paste0(y_train_faclev, "_pred_model_", p_stack_identifier)
            # debug02_ <- cbind(ypred_test_df, x_test_id)
            ypred_test_df <- cbind(ypred_test_df, id=x_test_id[, "id"])
            x_stack_test <- bind_rows(x_stack_test, ypred_test_df)
        }
        
    
        
    }  # end for loop
    
    
    # gather fold predictions into long format
    x_stack_train_folds_long <- tidyr::gather(x_stack_train_folds, key=feature_name, value=value, -id)
    
    # several test predictions for each ID, so average them within each ID 
    if(use_test) {
        
        # summarise multiple predictions per ID into a single prediction (mean of all)
        x_stack_test_mean <- x_stack_test %>%
            dplyr::group_by(id) %>% dplyr::summarise_all(funs(mean))
        
        # gather test_means into long format
        x_stack_test_long <- tidyr::gather(x_stack_test_mean, key=feature_name, value=value, -id)
        
    }
        
    
    
    # create a named list of the values to return from this function
    if(use_test) {
        return_list <- list(x_stack_train_folds_long = x_stack_train_folds_long,
                            x_stack_test_long = x_stack_test_long)
    } else {
        return_list <- list(x_stack_train_folds_long = x_stack_train_folds_long)
    }
    
    return(return_list)
    
    
}  # end function
    



    