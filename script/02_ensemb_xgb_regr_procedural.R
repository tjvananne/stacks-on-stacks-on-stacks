

#########################################
# Name space: x3
# target: petal_length
# type: regression
# model: XGBoost
# training data: 4 folds of train data
# pooling test data as additional feature
#' description: regression level1 model on sparse matrices using xgboost
#########################################



# initially available objects - these should be in global environment:

    # df_all        # all features in wide format
    # feats_all     # cols: id, feature_name, value
    # y             # cols: id, <target-var-name>



    # reset "button" for rapid dev -- delete all objects in global env except GBL configs and base data:
    GBLs <- ls()[grepl("^GBL_", ls())]
    rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))
    
    
    
    setdiff(names(df_all)[sapply(df_all, class) %in% c("numeric", "integer")], c("id", "dataset"))



# think of these like parameters / config settings
    x3_exp_number <- 3  # <-- this is the number of experiment within the stacks (x3 prefix means x3_epx_number is 3)
    x3_tar_var_ <- "petal_length"
    x3_cv_maxnrounds_ <- 1000
    x3_obj_func_ <- "reg:linear"
    x3_eval_metric_ <- "rmse"
    x3_train_ids_ <- df_all$id[df_all$dataset == 'train']
    x3_test_ids_ <- setdiff(df_all$id, x3_train_ids_)
    x3_cheaters_ <- c("plen_qt_1", "plen_qt_2", "plen_qt_3", "plen_qt_4")  # <-- thing that would leak this current target var into training data
    
        #' note about categorical variables in this "cheaters group", you must supply the name of 
        #' the variable AFTER it's value (category) has been concatenated to the name of the field.
    
    
    # retrain large, processing intensive files like bst_cv?
    x3_read_from_cache_ <- FALSE     
    
    
    
# DJ, stack that sh**!

    # x3_cheaters_ can't remove categorical variables if they already have their values
    # concatenated to them... Need to consider the best route to fix that.
    
    
    # remove NAs and cheater vars -- the only cheaters here is the target variable
    x3_all <- subset(feats_all, !is.na(value) & !feature_name %in% c(x3_tar_var_, x3_cheaters_))
    
    
    # create the feature map with the "all" object
    x3_all$feature_id <- as.numeric(as.factor(x3_all$feature_name))
    x3_all_feature <- x3_all[!duplicated(x3_all$feature_id), c("feature_id", "feature_name")] %>% 
        arrange(feature_id)
    
    
    # You can split these however you want, I'm just picking train/test split
    # one good option would be to make test all of the NA values, train all the valid values
    x3_train <- subset(x3_all, id %in% x3_train_ids_)
    x3_test <- subset(x3_all, id %in% x3_test_ids_)
    
    
    # split train and test, mark them, remove NAs
    x3_train$dataset <- 'train'
    x2_train <- subset(x3_train, !is.na(value))
    x3_test$dataset <- 'test'
    x3_test <- subset(x3_test, !is.na(value))
    
    
    # remove any features that aren't alike
    x3_feat_intersect_ <- dplyr::intersect(x3_train$feature_name, x3_test$feature_name)
    x3_train <- subset(x3_train, feature_name %in% x3_feat_intersect_)
    x3_test <- subset(x3_test, feature_name %in% x3_feat_intersect_)
    x3_all <- bind_rows(x3_train, x3_test)
    
    
    
    # numeric factorized id's need to be generated for train and test separately for sparseMatrix generation
    x3_train$id_num <- as.numeric(as.factor(x3_train$id))
    x3_test$id_num <- as.numeric(as.factor(x3_test$id)) 
    
    

    # create id mappings for train and test
    x3_train_id <- x3_train %>%
        select(id, id_num) %>%
        unique() %>%
        arrange(id_num)
    
    
    # join in this first level model's target variable (pretending like we don't have this data for "test")
    x3_all_y <- feats_all[feats_all$feature_name %in% x3_tar_var_, ]
    x3_train_id <- merge(x=x3_train_id, y=x3_all_y, by="id", all.x=T, all.y=F)
    
    
        # optional, mainly for debugging - create id / y mapping for test
        x3_test_id <- x3_test %>%
            select(id, id_num) %>%
            unique() %>% arrange(id_num)
        
        # only necessary if test has answers? or maybe not... it'll just be full of NA values at that point
        x3_test_id <- merge(x=x3_test_id, y=x3_all_y, by="id", all.x=T, all.y=F)
        rm(x3_all_y) 
        
        
    # data and labels are separated, make sure they are sorted at the same time in the same way
    x3_train <- x3_train %>% arrange(id_num)
    x3_train_id <- x3_train_id %>% arrange(id_num)
    y3_train <- x3_train_id$value
    
        
    # "i" and "j" are locations, not values. that is why we factored id's for train/test separately
    x3_train_sp <- sparseMatrix(
        i = x3_train$id_num,
        j = x3_train$feature_id,
        x = x3_train$value
    )
     
    
            # relationship between these attributes of these objects is critical to understanding what's happening
            nrow(x3_train_sp); length(unique(x3_train$id)); max(x3_train$id_num); length(y3_train)
    
    
    # when separating data, make sure they are both sorted the same way
    x3_test <- x3_test %>% arrange(id_num)
    x3_test_id <- x3_test_id %>% arrange(id_num)
    y3_test <- rep(NA, nrow(x3_test_id))  # <-- is this necessary ?        
            
    
    
    # test sparse matrix
    x3_test_sp <- sparseMatrix(
        i = x3_test$id_num,
        j = x3_test$feature_id,
        x = x3_test$value
    )
    
            # once again, make sure you understand this:
            nrow(x3_test_sp); length(unique(x3_test$id)); max(x3_test$id_num); length(y3_test)        
    
    
            
# looped manual CV
            
    # fold creation
    folds <- 4
    x3_cv <- caret::createFolds(1:nrow(x3_train_sp), k = folds)
    
    # collectors
    x3_stack_train <- data.frame()
    x3_stack_test <- data.frame()  # <-- going to pool these and then bag them by ID
    
    
    for(i in 1:folds) {
        
        # i <- 1
        # isolate fold indexes for this iteration
        fold_indx <- x3_cv[[i]]
        
        # create IN FOLD DMat from sparse train matrix
        x3_train_fold_id_ <- x3_train_id[fold_indx, ]
        x3_train_fold_sp_ <- x3_train_sp[fold_indx, ]
        
            # for debugging
            x3_train_id[fold_indx, ]
            
        # create OUT OF FOLD DMats
        x3_train_sp_ <- x3_train_sp[-fold_indx, ]
        y3_train_ <- (y3_train[-fold_indx ] - 1)
            
        
        # DMats
        dx3_fold_ <- xgb.DMatrix(x3_train_fold_sp_)          # in fold, this is our "test" for this cv iter
        dx3_ <- xgb.DMatrix(x3_train_sp_, label = y3_train_) # out of fold, this is our train
        dx3_test_ <- xgb.DMatrix(x3_test_sp)
        
            # fyi -- troubleshooting
            dim(dx3_fold_)
            dim(dx3_)
        
        # xgboost parameters for CV and xgbtrain
        param <- list("objective" = x3_obj_func_,
                      "eval_metric" = x3_eval_metric_,
                      "eta" = 0.01,              # learning rate, how far to step in gradient descent movements
                      "max_depth" = 6,
                      "subsample" = 0.7,
                      "colsample_bytree" = 0.3,
                      # "lambda" = 1.0,
                      "alpha" = 1.0,             # L1 regularization
                      # "min_child_weight" = 6,  
                      # "gamma" = 10,            # complexity control (minimum impact on loss function for split to occur)
                      "nthread" = 6)             # CPUs to use
            
            
        # small datasets we could cv every time?
        if(i == 1) {
            # cv is to find optimal "nrounds"
            
            bst_cv_ = xgb.cv(param = param,
                            data = dx3_,   # out of fold data
                            nfold = folds,
                            nrounds = x3_cv_maxnrounds_,
                            early_stop_round = 10)
            
            # cache here if this takes a long time
            saveRDS(bst_cv_, file.path(GBL_cache_path, "x3_bst_xgb_cv.rds"))
        }
        
        
        x3_eval_log_ <- data.frame(bst_cv_$evaluation_log)
        x3_eval_log_col_ <- paste0('test_', x3_eval_metric_, '_mean')
        x3_bst_nrounds_ <- which.min( as.numeric(x3_eval_log_[[x3_eval_log_col_]]))
        
        
        # train model:
        xgb3_ <- xgboost(data = dx3_,  # <-- training on out-of-fold DMat
                         param = param,
                         nround = x3_bst_nrounds_,
                         save_period = NULL)
            
        if(i == 1) {
            xgbimp_ <- xgb.importance(feature_names = unique(x3_train$feature_name), model=xgb3_)
            xgboost::xgb.ggplot.importance(xgbimp_)
        }
        
        
        
        
        
        # predict on the fold 
        y3_train_fold_preds_ <- predict(xgb3_, dx3_fold_)
        # y3_fold_df <- as.data.frame(y3_fold_mat, stringsAsFactors = F)
        
            # debugging
            inspect_fold_preds <-  cbind(x3_train_fold_id_, y3_train_fold_preds_)
            # yes, it looks like classes maintain the same order
        
        
        # combine train fold data with y-preds fold
        y3_fold_df <- cbind(x3_train_fold_id_, y3_train_fold_preds_)    
        names(y3_fold_df)[grepl("^y3_train_fold_preds_$", names(y3_fold_df))] <- 
            paste0(x3_tar_var_, "_preds_", x3_exp_number)
       
            
        
        # predict on test
        dim(dx3_test_)
        y3_test_preds_ <- predict(xgb3_, dx3_test_)
        y3_test_preds_df_ <- cbind(x3_test_id, y3_test_preds_)
        
        
            # debugging
            inspect_test_preds <- cbind(x2_test_id, y2_test_df)
        
            
        names(y2_test_df) <- paste0(x2_tar_var_, '_preds_', 1:length(x2_possible_tar_vals_))
        y2_test_df <- cbind(y2_test_df, id=x2_test_id[, c("id")])
        x2_stack_test <- bind_rows(x2_stack_test, y2_test_df)
        
    }
    
        
# gather our in-fold predictions into long format
x2_stack_train_gather <- tidyr::gather(x2_stack_train, key=feature_name, value=value, -id)


# "bag" each fold's version of our test predictions
x2_stack_test <- x2_stack_test %>%
    dplyr::group_by(id) %>%
    dplyr::summarise_all(funs(mean))
    

# gather the test preds into long format
x2_stack_test_gather <- tidyr::gather(x2_stack_test, key=feature_name, value=value, -id)


# predicting on 3 classes, so 150 total unique IDs having 3 features each gives us 450
x2_stack_all <- bind_rows(x2_stack_train_gather, x2_stack_test_gather)
print(dim(x2_stack_all))


# cache / writeout here
saveRDS(x2_stack_train, file.path(GBL_cache_path, "x2_stack_train.rds"))
saveRDS(x2_stack_test, file.path(GBL_cache_path, "x2_stack_test.rds"))


# clean up
GBLs <- ls()[grepl("^GBL_", ls())]
rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))




