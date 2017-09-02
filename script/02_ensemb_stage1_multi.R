

#########################################
# Name space: x2
# target: species
# type: multi-class classification
# model: XGBoost
# training data: 4 folds of train data
# pooling test data as additional feature
#########################################


# initially available objects - these should be in global environment:

    # df_all        # all features in wide format
    # feats_all     # cols: id, feature_name, value
    # y             # cols: id, <target-var-name>



    # reset "button" for rapid dev -- delete all objects in global env except GBL configs and base data:
    GBLs <- ls()[grepl("^GBL_", ls())]
    rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))




# think of these like parameters / config settings

    x2_tar_var_ <- "species"
    x2_cv_maxnrounds_ <- 1000
    x2_obj_func_ <- "multi:softprob"
    x2_eval_metric_ <- "mlogloss"
    x2_train_ids_ <- df_all$id[df_all$dataset == 'train']
    x2_test_ids_ <- setdiff(df_all$id, x2_train_ids_)
    
    # retrain large, processing intensive files like bst_cv?
    x2_read_from_cache_ <- FALSE     
    
    
    
# DJ, stack that sh**!

    # identify all possible values of this stackers target variable
    x2_possible_tar_vals_ <- paste0(x2_tar_var_, "_", levels(df_all[, x2_tar_var_]))
    x2_all <- subset(feats_all, (!is.na(value) & !feature_name %in% x2_possible_tar_vals_))
    
    
    # create the feature map with the "all" object
    x2_all$feature_id <- as.numeric(as.factor(x2_all$feature_name))
    x2_all_feature <- x2_all[!duplicated(x2_all$feature_id), c("feature_id", "feature_name")] %>% 
        arrange(feature_id)
    
    
    # You can split these however you want, I'm just picking train/test split
    # one good option would be to make test all of the NA values, train all the valid values
    x2_train <- subset(x2_all, id %in% x2_train_ids_)
    x2_test <- subset(x2_all, id %in% x2_test_ids_)
    
    
    # split train and test, mark them, remove NAs
    x2_train$dataset <- 'train'
    x2_train <- subset(x2_train, !is.na(value))
    x2_test$dataset <- 'test'
    x2_test <- subset(x2_test, !is.na(value))
    
    
    # remove any features that aren't alike
    x2_feat_intersect_ <- dplyr::intersect(x2_train$feature_name, x2_test$feature_name)
    x2_train <- subset(x2_train, feature_name %in% x2_feat_intersect_)
    x2_test <- subset(x2_test, feature_name %in% x2_feat_intersect_)
    x2_all <- bind_rows(x2_train, x2_test)
    
    
    
    # numeric factorized id's need to be generated for train and test separately for sparseMatrix generation
    x2_train$id_num <- as.numeric(as.factor(x2_train$id))
    x2_test$id_num <- as.numeric(as.factor(x2_test$id)) 
    
    

    # create id mappings for train and test
    x2_train_id <- x2_train %>%
        select(id, id_num) %>%
        unique() %>%
        arrange(id_num)
    
    
    # join in this first level model's target variable (pretending like we don't have this data for "test")
    x2_all_y <- feats_all[feats_all$feature_name %in% x2_possible_tar_vals_, c("id", "feature_name")]
    x2_train_id <- merge(x=x2_train_id, y=x2_all_y, by="id", all.x=T, all.y=F)
    x2_train_id$this_target <- as.numeric(as.factor(x2_train_id$feature_name))
    
    
        # optional, mainly for debugging - create id / y mapping for test
        x2_test_id <- x2_test %>%
            select(id, id_num) %>%
            unique() %>% arrange(id_num)
        
        # only necessary if test has answers? or maybe not... it'll just be full of NA values at that point
        x2_test_id <- merge(x=x2_test_id, y=x2_all_y, by="id", all.x=T, all.y=F)
        rm(x2_all_y) 
        
        
    # data and labels are separated, make sure they are sorted at the same time in the same way
    x2_train <- x2_train %>% arrange(id_num)
    x2_train_id <- x2_train_id %>% arrange(id_num)
    y2_train <- x2_train_id$this_target
    
        
    # "i" and "j" are locations, not values. that is why we factored id's for train/test separately
    x2_train_sp <- sparseMatrix(
        i = x2_train$id_num,
        j = x2_train$feature_id,
        x = x2_train$value
    )
     
    
            # relationship between these attributes of these objects is critical to understanding what's happening
            nrow(x2_train_sp); length(unique(x2_train$id)); max(x2_train$id_num); length(y2_train)
    
    
    # when separating data, make sure they are both sorted the same way
    x2_test <- x2_test %>% arrange(id_num)
    x2_test_id <- x2_test_id %>% arrange(id_num)
    y2_test <- rep(NA, nrow(x2_test_id))  # <-- is this necessary ?        
            
    
    
    # test sparse matrix
    x2_test_sp <- sparseMatrix(
        i = x2_test$id_num,
        j = x2_test$feature_id,
        x = x2_test$value
    )
    
            # once again, make sure you understand this:
            nrow(x2_test_sp); length(unique(x2_test$id)); max(x2_test$id_num); length(y2_test)        
    
    
            
# looped manual CV
            
    # fold creation
    folds <- 4
    x2_cv <- caret::createFolds(1:nrow(x2_train_sp), k = folds)
    
    # collectors
    x2_stack_train <- data.frame()
    x2_stack_test <- data.frame()  # <-- going to pool these and then bag them by ID
    
    
    for(i in 1:folds) {
        
        # i <- 1
        # isolate fold indexes for this iteration
        fold_indx <- x2_cv[[i]]
        
        # create IN FOLD DMat from sparse train matrix
        x2_train_fold_id_ <- x2_train_id[fold_indx, ]
        x2_train_fold_sp_ <- x2_train_sp[fold_indx, ]
        
            # for debugging
            x2_train_id[fold_indx, ]
            
        # create OUT OF FOLD DMats
        x2_train_sp_ <- x2_train_sp[-fold_indx, ]
        y2_train_ <- (y2_train[-fold_indx ] - 1)
            
        
        # DMats
        dx2_fold_ <- xgb.DMatrix(x2_train_fold_sp_)          # in fold, this is our "test" for this cv iter
        dx2_ <- xgb.DMatrix(x2_train_sp_, label = y2_train_) # out of fold, this is our train
        dx2_test_ <- xgb.DMatrix(x2_test_sp)
        
            # fyi -- troubleshooting
            dim(dx2_fold_)
            dim(dx2_)
        
        # xgboost parameters for CV and xgbtrain
        param <- list("objective" = x2_obj_func_,
                      "eval_metric" = x2_eval_metric_,
                      "num_class" = length(x2_possible_tar_vals_),
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
                            data = dx2_,   # out of fold data
                            nfold = folds,
                            nrounds = x2_cv_maxnrounds_,
                            early_stop_round = 10)
            
            # cache here if this takes a long time
            saveRDS(bst_cv_, file.path(GBL_cache_path, "x2_bst_xgb_cv.rds"))
        }
        
        bst_nrounds2_ <- which.min(bst_cv_$evaluation_log$test_mlogloss_mean)
        
        
        # train model:
        xgb2_ <- xgboost(data = dx2_,  # <-- training on out-of-fold DMat
                         param = param,
                         nround = bst_nrounds2_,
                         save_period = NULL)
            
        if(i == 1) {
            xgbimp_ <- xgb.importance(feature_names = unique(x2_train$feature_name), model=xgb2_)
            xgboost::xgb.ggplot.importance(xgbimp_)
        }
        
        
        
        
        
        # predict on the fold 
        y2_train_fold_ <- predict(xgb2_, dx2_fold_)
        y2_fold_mat <- matrix(y2_train_fold_, nrow(x2_train_fold_sp_), ncol=length(x2_possible_tar_vals_), byrow=T)
        y2_fold_df <- as.data.frame(y2_fold_mat, stringsAsFactors = F)
        
            # debugging
            inspect_fold_preds <-  cbind(x2_train_id[fold_indx,], y2_fold_df)
            # yes, it looks like classes maintain the same order
        
        # if you plan on using this as a feature as well as the already present "species_<species>" then 
        # we need to add "_preds_" or some other indicator in there that this has been stacked
        names(y2_fold_df) <- paste0(x2_tar_var_, '_preds_', 1:length(x2_possible_tar_vals_))
        y2_fold_df <- cbind(y2_fold_df, id=x2_train_id[fold_indx, c('id')])
        x2_stack_train <- bind_rows(x2_stack_train, y2_fold_df)
            
        
        # predict on test
        dim(dx2_test_)
        y2_test_ <- predict(xgb2_, dx2_test_)
        y2_test_mat <- matrix(y2_test_, nrow(dx2_test_), ncol=length(x2_possible_tar_vals_), byrow=T)    
        y2_test_df <- as.data.frame(y2_test_mat, stringsAsFactors = F)
        
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




