

#' vision for this project:
#' You provide a data.frame of features and this thing does the rest. Will separate out 
#' and preprocess binary categorical variables, multi-categorical variables, numeric, etc.
#' 
#' 1st step is to produce the explicit / procedural version, then go back and functionalize

# predict "sepal_width" from iris dataset using 3 numeric fields and one categorical
# utilize stacked generalization ensembling


# set working directory
setwd("C:/Users/tvananne/Documents/testbed/ensembling_stacking_combining_models/airbnb_kaggle_example_smallest/script")
getwd()


# source in scripts
source('ensemb_utils_small.R')





# light config -------------------------
    train_size <- 0.7  # between 0.1 - 0.9



# load data ----------------------------
    
    set.seed(1)
    data_len <- nrow(iris)
    myiris <- iris[sample(1:data_len, data_len), ]
    names(myiris) <- gsub("\\.", "_", tolower(names(myiris)))
    
    
    
    # generate rand character string ids:
    randchar_pool_ <- c(letters, 0:9)
    myiris$id <- replicate(
        n=nrow(myiris), 
        expr = paste0(randchar_pool_[sample(1:length(randchar_pool_), 10)], collapse='')
    )
    
    # should be TRUE (will always be true if you leave set.seed(1) alone)
    sum(duplicated(myiris$id)) == 0
    

# data preproc -------------------------

    
    
    # will be labels for both train and test
    y <- myiris[, c("id", "sepal_width")]  # immediately after generating "id" field
    myiris$sepal_width <- NULL
    
    
    
    # train test split
    train_indx <- sample(1:data_len, (train_size * data_len))
    train <- myiris[train_indx, ]
    test <- myiris[setdiff(1:data_len, train_indx), ]
    train$dataset <- 'train'
    test$dataset <- 'test'
    myiris <- bind_rows(train, test)
    
    y_train <- y[train_indx, ]
    y_test <- y[setdiff(1:data_len, train_indx), ]  # pretend like we don't have this if it's "new data"
    


# Additional feature generation and data cleansing -----------
    
    # let's create a few more categorical variables so we have more than just 1

    
    # all categorical variables should have their values attached to the feature name
    # myiris$species <- as.factor(paste0("species_", as.character(myiris$species)))  # no.. do this at stack
    
    
    # split petal_length into categorical (factor) quartile bins
    plen_ <- myiris$petal_length
    plen_qt_int_ <- as.integer(cut2(plen_, c(
                                               min(plen_, na.rm=T), 
                                               quantile(plen_, 0.25, na.rm=T), 
                                               quantile(plen_, 0.50, na.rm=T), # aka, median 
                                               quantile(plen_, 0.75, na.rm=T))))
    myiris$plen_qt <- as.factor(plen_qt_int_)
    
    
    # split petal_width into binary categories (< median, > median)
    pwid_ <- myiris$petal_width
    summary(pwid_)
    pwid_bin_int_ <- as.integer(cut2(pwid_, c(
                                           min(pwid_, na.rm=T),
                                           quantile(pwid_, 0.50, na.rm=T))))
    myiris$pwid_bin <- as.factor(pwid_bin_int_)

    
    
    # remove items ending with underscore in current global env
    rm(list = ls()[grepl("_$", ls())])
    
        
# stack all features -------------

    # GOAL, stack into [id, feature_name, value] 
    
    
    # numerics and categoricals should be very clearly identified by the time you reach this point
    sapply(myiris, class)
    
    
    # stack numerical features together; then stack categorical features together
    feats_name_num <- setdiff(names(myiris)[sapply(myiris, class) %in% c("numeric", "integer")], c("id", "dataset"))
    feats_name_cat <- setdiff(names(myiris)[sapply(myiris, class) %in% c("character", "factor")], c("id", "dataset"))
    
    
    # this can handle ALL numeric fields
    # gather all numeric features into feature_name / column format
    feats_num <- gather(myiris[, c(feats_name_num, "id")], key=feature_name, value=value, -id) %>%
        arrange(id)
    
    
    # writing loop to handle ALL feature / categoricals
    feats_cat <- data.frame()
    for(feat in feats_name_cat) {
        
        print(feat)
        
        
        # isolate this feat
        df_feats_ <- myiris[, c("id", feat)]
        assign(x = paste0("pos_values_", feat), value = as.character(unique(df_feats_[, 2])))
        df_feats_[, 2] <- paste0(feat, "_", as.character(df_feats_[, 2]))
        df_feats_$value <- 1
        names(df_feats_) <- c("id", "feature_name", "value")
        feats_cat <- bind_rows(feats_cat, df_feats_)
        
    }
    
    
    
    # combine numeric and categorical features
    feats_all <- bind_rows(feats_num, feats_cat)
    
    
    # optional cache right here!
    list.files('../cache/')
    saveRDS(feats_all, '../cache/feats_all.rds')
    
    
    # explore this another time....
    # h5::createDataSet(data = as.matrix(feats_all), datasetname = '../cache/feats_all.h5')
    
    # remove items ending with underscore in current global env
    rm(list = ls()[grepl("_$", ls())])
    


# stacking begins here -----------------------------------

#########################################
# target: species
# model: XGBoost
# training data: 3 folds of train data and then test data 
#########################################

    
        #' Take-away: If there were a subset of records that had bad data for
        #' their species feature, then we could force THOSE records into
        #' being the "test" dataset that we don't have good answers for, then
        #' train the algo on folds of the ones we DO have good data for, then 
        #' just use the predictions from the "test" dataset as feature for 
        #' the next level 
        #' 
        #' In this experiment, we have good data for all records in the
        #' species field, but let's pretend like the "test" data doesn't 
        #' have this field. 
    
    
    # if using the cache
    # feats_all <- readRDS('../cache/feats_all.rds')
    
    
    x2_tarvar <- "species"
    
    
    # can we break the dependency on having this pos_values_species object?
    # maybe use this instead?:
    levels(myiris$species)
    x2_posvals <- paste0(x2_tarvar, "_", levels(myiris[,'species'])) # <- look for these for our "y2"
    
        # re-evaluate the number of possible values in target variable after doing the 
        # analysis of the intersection between features in train/test
    
    
    cv_maxnrounds_ <- 1000
    
    # determine automagically what type of xgboost model we need
    multi_ <- length(x2_posvals) > 2 
    if(multi_) {
        xgb2_obj_ <- "multi:softprob"
        x2_eval_metric <- "mlogloss"
    } else {
        xgb2_obj_ <- ""
        x2_eval_metric <- ''
    }
    
    
    #' Still haven't done any checks for which features can be found in which datasets
    #' Need to find the intersection between those features and remove any that don't
    #' have a match in the opposite (test) dataset
    #' 
    #' Also, removal of NAs and blank values or something
    
    
    
    # X2_all -- UPDATE, now removing the "cheater" variable
    x2_all <- subset(feats_all, !is.na(value) & !feature_name %in% x2_posvals)
    y2_all <- subset(feats_all, !is.na(value) & feature_name %in% x2_posvals)
    
    # create the feature map with the "all" object
    x2_all$feature_id <- as.numeric(as.factor(x2_all$feature_name))
    x2_all_feature <- x2_all[!duplicated(x2_all$feature_id), c("feature_id", "feature_name")] %>% 
        arrange(feature_id)
    
    
    # Keep in mind, you can split train and test in these lower level models however you want
    # I chose train and test because it's simple, A good way to decide is to pick the records
    # with high density / cleanliness in this model's target variable
    x2_train <- subset(x2_all, id %in% y_train$id)
    x2_test <- subset(x2_all, id %in% y_test$id)
    x2_train$dataset <- 'train'
    x2_test$dataset <- 'test'
    x2_all <- bind_rows(x2_train, x2_test)
    
    
    
    # numeric factorized id's need to be generated for train and test separately for sparseMatrix generation
    x2_train$id_num <- as.numeric(as.factor(x2_train$id))
    x2_test$id_num <- as.numeric(as.factor(x2_test$id)) 
    
    
        # dummmy holding commented stuff
        dummy <- function(x) {
                # I change my mind, I don't think this is important? keeping it commented in case I want to revisit..
                    # # if the target variable is in both train and test... you'd follow this path here:
                    # x2_all <- bind_rows(x2_train, x2_test)
                    # 
                    # x2_all_id <- x2_all %>%
                    #     select(id, id_num) %>%
                    #     unique() %>%
                    #     arrange(id_num)
                    # 
                    # x2_all_id <- merge(x=x2_all_id, feats_all[feats_all$feature_name %in% x2_posvals, c("id", "feature_name")])
           
        }
        
        
    #' id_num ARE NOT UNIQUE IDENTIFIERS, just the numeric factor order of the unique id field
    #' within train and test respectively


    # create id mappings for train and test
    x2_train_id <- x2_train %>%
        select(id, id_num) %>%
        unique() %>%
        arrange(id_num)
    
    
    # join in this first level model's target variable (pretending like we don't have this data for "test")
    x2_train_id <- merge(x=x2_train_id, 
                         y=feats_all[feats_all$feature_name %in% x2_posvals, c("id", "feature_name")], 
                         by="id", all.x=T, all.y=F)
    x2_train_id$this_target <- as.numeric(as.factor(x2_train_id$feature_name))
    
    
        # is this necessary? -- yes, for debugging
        x2_test_id <- x2_test %>%
            select(id, id_num) %>%
            unique() %>% arrange(id_num)
        
        # you might not always be able to do this (below) but I'm doing it for debugging
        # might now always have access to the answer in the "test" set, especially for new data
        x2_test_id <- merge(x=x2_test_id,
                            y=feats_all[feats_all$feature_name %in% x2_posvals, c("id", "feature_name")],
                            by="id", all.x=T, all.y=F)
        
        
    # when separating training data from labels, make sure they are sorted the same immediately prior to separation
    x2_train <- x2_train %>% arrange(id_num)
    x2_train_id <- x2_train_id %>% arrange(id_num)
    y2_train <- x2_train_id$this_target
    
        
    # read the docs on what "i" and "j" mean here, they are locations, not values themselves
    x2_train_sp <- sparseMatrix(
        i = x2_train$id_num,
        j = x2_train$feature_id,
        x = x2_train$value
    )
     
    
            # relationship between these attributes of these objects is critical to understanding what's happening
            nrow(x2_train_sp)
            length(unique(x2_train$id))
            max(x2_train$id_num)
            length(y2_train)
    
    
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
            nrow(x2_test_sp)
            length(unique(x2_test$id))
            max(x2_test$id_num)
            length(y2_test)        
    
    
    # fold creation
    folds <- 4
    x2_cv <- caret::createFolds(1:nrow(x2_train_sp), k = folds)
    
    # collectors
    x2_stack <- data.frame()
    x2_stack_test <- data.frame()  # <-- going to pool these and then bag them by ID
    
    
    for(i in 1:folds) {
        
        # in-loop implementation stuff will end in "_" underscore
        
        # for each fold... create x2_fold_
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
        
            
        # can we incorporate TPOT here once we start getting really good at this?
        param <- list("objective" = xgb2_obj_,
                      "eval_metric" = x2_eval_metric,
                      "num_class" = length(x2_posvals),
                      "eta" = 0.01,
                      "max_depth" = 6,
                      "subsample" = 0.7,
                      "colsample_bytree" = 0.3,
                      # "lambda" = 1.0,
                      "alpha" = 1.0,
                      # "min_child_weight" = 6,
                      # "gamma" = 10,
                      #"nthread" = 24)
                      "nthread" = 6)
            
        # small datasets we could cv every time?
        if(i == 1) {
            # cv is to find optimal "nrounds"
            
            bst_cv_ = xgb.cv(param = param,
                            data = dx2_,   # out of fold data
                            nfold = folds,
                            nrounds = cv_maxnrounds_,
                            early_stop_round = 10)
            
            # cache here if this takes a long time
        }
        
        bst_nrounds2_ <- which.min(bst_cv_$evaluation_log$test_mlogloss_mean)
        
        
        # train model:
        xgb2_ <- xgboost(data = dx2_,  # <-- training on out-of-fold DMat
                         param = param,
                         nround = bst_nrounds2_)
            
        if(i == 1) {
            xgbimp_ <- xgb.importance(feature_names = unique(x2_train$feature_name), model=xgb2_)
            xgboost::xgb.ggplot.importance(xgbimp_)
        }
        
        
        
        
        
        # predict on the fold 
        y2_train_fold_ <- predict(xgb2_, dx2_fold_)
        y2_fold_mat <- matrix(y2_train_fold_, nrow(x2_train_fold_sp_), ncol=length(x2_posvals), byrow=T)
        y2_fold_df <- as.data.frame(y2_fold_mat)
        
            # debugging
            inspect_fold_preds <-  cbind(x2_train_id[fold_indx,], y2_fold_df)
            # yes, it looks like classes maintain the same order
        
        # if you plan on using this as a feature as well as the already present "species_<species>" then 
        # we need to add "_preds_" or some other indicator in there that this has been stacked
        names(y2_fold_df) <- paste0(x2_tarvar, '_preds_', 1:length(x2_posvals))
        y2_fold_df <- cbind(y2_fold_df, id=x2_train_id[fold_indx, c('id')])
        x2_stack <- bind_rows(x2_stack, y2_fold_df)
            
        
        # predict on test
        dim(dx2_test_)
        y2_test_ <- predict(xgb2_, dx2_test_)
        y2_test_mat <- matrix(y2_test_, nrow(dx2_test_), ncol=length(x2_posvals), byrow=T)    
        y2_test_df <- as.data.frame(y2_test_mat)
        
            # debugging
            inspect_test_preds <- cbind(x2_test_id, y2_test_df)
        
            
        names(y2_test_df) <- paste0(x2_tarvar, '_preds_', 1:length(x2_posvals))
        y2_test_df <- cbind(y2_test_df, id=x2_test_id[, c("id")])
        x2_stack_test <- bind_rows(x2_stack_test, y2_test_df)
        
    }
    
    
    
        # inspection
        
        length(unique(x2_stack$id))
        dim(x2_stack)
        nrow(x2_stack)
        
        
        
        dim(x2_stack_test)
        nrow(x2_stack_test) / folds  # number of rows of original test data
        
        
        
        
        
# gather our in-fold predictions into long format
x2_stack_gather <- tidyr::gather(x2_stack, key=feature_name, value=value, -id)


# "bag" each fold's version of our test predictions
x2_stack_test <- x2_stack_test %>%
    dplyr::group_by(id) %>%
    dplyr::summarise_all(funs(mean))
    

# gather the test preds into long format
x2_stack_test_gather <- tidyr::gather(x2_stack_test, key=feature_name, value=value, -id)



