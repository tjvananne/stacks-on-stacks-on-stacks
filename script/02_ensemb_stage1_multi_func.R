

# 02_ensemb_stage1_multi_function


# FUNC PARAM SETUP ------------------------------------------------------

# pass this in as a parameter itself (the list)
params <- list("objective" = "multi:softprob",
               "eval_metric" = "mlogloss",
               "num_class" = 3,
               "eta" = 0.01,              
               "max_depth" = 6,
               "subsample" = 0.7,
               "colsample_bytree" = 0.3,
               # "lambda" = 1.0,
               "alpha" = 1.0,             
               # "min_child_weight" = 6,  
               # "gamma" = 10,            
               "nthread" = 6)     



# cheaters should be based on feature_name within feats_all, NOT within df_all
these_cheaters <- feats_all$feature_name[grepl("^species_", feats_all$feature_name)] %>% unique()


# set up y values for THIS stack, not for overall project
this_stack_y <- df_all[, c("id", "species")]
this_stack_y <- paste0("species_", this_stack_y$species)



# FUNC DEF ---------------------------------------------------------------

    #' Note: this function assumes you have a dataframe in your global environment named "feats_all" and
    #' that dataframe has the following columns:
    #'   - id: unique identifier for an observation in your data
    #'   - feature_name: the name of a specific feature
    #'   - value: the value that corresponds to the intersection of the id and feature_name in the data
    #'   
    #'   
    #' You should also have an object called "y" in your environment with the following columns:
    #'   - id: unique identifier for an observation in your data (same as above)
    #'   - <target variable name>: this will obviously vary
    #'   
    #'   
    #'   We're all consenting adults here. Understand the constraints of this function and don't 
    #'   point the loaded gun at your own foot.
    #'   
    #'   Let's try it without passing it explicitly first, then try it with and compare the RAM usage



layer1_multi <- function(
    # p_feats_all,  # <-- risky, but let's just use the global env version...
    p_stack_y,      # <-- the y's relative to THIS stack, not for the overall project
    p_tar_var,      # <- "species"
    p_train_ids,    # <- df_all$id[df_all$dataset == 'train']
    p_test_ids,     # <- setdiff(df_all$id, x2_train_ids_)
    p_xgb_params,
    p_cv_folds,
    p_cv_rounds,
    p_cv_earlystop,
    p_cheaters,     # <- c("")
    p_read_from_cache = FALSE # <- FALSE     
) {
    
    
    # for testing, we want to just have the parameter values available to us interactively
    p_tar_var <- "species"
    p_train_ids <- df_all$id[df_all$dataset == 'train']
    p_test_ids  <- df_all$id[df_all$dataset == 'test']
    p_xgb_params <- params
    p_cv_folds <- 4
    p_cv_rounds <- 3000
    p_cv_earlystop <- 15
    p_cheaters <- these_cheaters
    p_read_from_cache = FALSE
    
    
    print("Initializing classification stage 1 stacker...")
    
    
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
    x_test <- subset(x_all, id %in% p_test_ids)    
    x_test$dataset <- 'test'
    x_test <- subset(x_test, !is.na(value))
    
    # feature intersection between groups
    x_feat_intersect <- dplyr::intersect(x_train$feature_name, x_test$feature_name)
    x_train <- subset(x_train, feature_name %in% x_feat_intersect)    
    x_test <- subset(x_test, feature_name %in% x_feat_intersect)
    x_all <- dplyr::bind_rows(x_train, x_test)
    
    # numeric factorized id's generated within each group -- NOT UNIQUE IDENTIFIERS
    x_train$id_num <- as.numeric(as.factor(x_train$id))
    x_test$id_num <- as.numeric(as.factor(x_test$id))
    
    # create id mappings for train and test separately
    x_train_id <- x_train %>% select(id, id_num) %>% unique() %>% arrange(id_num)
    x_test_id <- x_test %>% select(id, id_num) %>% unique() %>% arrange(id_num)
    
    # map the answers to each id set
    # y_train <- 
    # y_test <- x_test_id$this_target
    
    x_train_id <- merge(x=x_train_id, y=y, by="id", all.x=T, all.y=F)
    x_test_id <- merge(x=x_test_id, y=y, by="id", all.x=T, all.y=F)    
    names(x_train_id) <- c("id", "id_num", "this_target")
    names(x_test_id) <- c("id", "id_num", "this_target")
    
    # arrange all of these at the same time before generating sparse matrices
    x_train <- x_train %>% arrange(id_num)
    x_train_id <- x_train_id %>% arrange(id_num)
    x_test <- x_test %>% arrange(id_num)
    x_test_id <- x_test_id %>% arrange(id_num)
    
    
    # generate sparse matrices
    x_train_sp <- sparseMatrix(i = x_train$id_num, j = x_train$feature_id, x = x_train$value)
    x_test_sp <- sparseMatrix(i = x_test$id_num, j = x_test$feature_id, x = x_test$value)
    
    
        # assertions to keep the function from silent errors
        assert_that(nrow(x_train_sp) == nrow(x_train_id) & 
                    nrow(x_train_sp) == length(unique(x_train$id)) & 
                    length(y_train) == nrow(x_train_sp))
        
        assert_that(nrow(x_test_sp) == nrow(x_test_id) & 
                    nrow(x_test_sp) == length(unique(x_test$id)) &
                    length(y_test) == nrow(x_test_sp))
    
    
    # manual cross validation
    gc()
    print("Begin manual cross validation...")
    x_cv <- caret::createFolds(1:nrow(x_train_sp), k = p_cv_folds)
    
    # collectors
    x2_stack_train <- data.frame()
    x2_stack_test <- data.frame()  # <-- going to pool these and then bag them (average) by ID
    
    # dmat for test can 
    dx_test <- xgb.DMatrix(x_test_sp)
    
    for(i in 1:p_cv_folds) {
        
        fold_indx <- x_cv[[i]]
        
        # create in-fold train data
        x_train_fold_id_ <- x_train_id[fold_indx, ]
        x_train_fold_sp_ <- x_train_sp[fold_indx, ]
        dx_train_fold_ <- xgb.DMatrix(x_train_fold_sp_)
        
        # create out-of-fold train data
        x_train_sp_ <- x_train_sp[-fold_indx, ]
        y_train_ <- y_train[-fold_indx]
        dx_train_ <- xgb.DMatrix(x_train_sp_, label = y_train_)
        
            # for debugging:
            # x_train_id[fold_indx, ]
        
        if(i == 1) {
            
            # cv to find optimal number of rounds
            bst_cv_ <- xgb.cv(params=params, data=dx_train_, nfold=p_cv_folds, 
                              nrounds=p_cv_rounds, early_stopping_rounds = p_cv_earlystop)
            
        }
        
              
    }    
        
    
    
}
    



# TEST FUNC ---------------------------------------------------------------

    
    
    # actual function call
    returned_thing <- layer1_multi(
                 # p_feats_all = feats_all,
                 p_tar_var = "species",
                 p_train_ids = df_all$id[df_all$dataset == 'train'],
                 p_test_ids = df_all$id[df_all$dataset == 'test'],
                 p_xgb_params = params,
                 p_cheaters = these_cheaters,
                 p_read_from_cache = FALSE)


