

# testing the xgboost sparse matrix classification function for stage 1-n stackers:

# 1) load in data and run the preprocessing script from main
# 2) source in the function
# 3) begin tests:
#    A) Predict on species using all data (no test data specified) and read from cache
#    B) Predict on species using all data (no test data specified) and don't read from cache
#    C) Predict on species using train data for train, test data for test, read from cache
#    D) Predict on species using train data for train, test data for test, don't read from cache
#    z) Predict on 

# need to finish building these tests



# FROM MAIN() ----------------------------------------------------------------------

    # set working directory (based on which machine I'm currently using)
    if(Sys.info()[["sysname"]] == "Windows" & Sys.info()[["user"]] == "Taylor") {
        setwd("C:/Users/Taylor/Documents/Nerd/github_repos/stacks_on_stacks/script")
    } else {
        setwd("C:/Users/tvananne/Documents/personal/github/stacks_on_stacks/script")
    }; getwd()
    
    
    # source in utilities script
    source('zz_ensemb_utils.R')  # <-- this should not remove items from environment
    
    
    if(TRUE) {
        source("01_ensemb_sparse_preproc.R")  # <-- standardized rules for preproc
        # complex datasets will have several preproc / first stage feature generation files
    }




# TEST A ---------------------------------------------------------------

    # pass this in as a parameter itself (the list)
    params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = 3,
                   "eta" = 0.01, "max_depth" = 6, "subsample" = 0.7, "colsample_bytree" = 0.3,
                   # "lambda" = 1.0, # "min_child_weight" = 6, # "gamma" = 10,
                   "alpha" = 1.0, "nthread" = 6)     
    
    
    # cheaters should be based on feature_name within feats_all, NOT within df_all
    these_cheaters <- feats_all$feature_name[grepl("^species_", feats_all$feature_name)] %>% unique()
    
    
    # set up y values for THIS stack, not for overall project
    this_stack_y <- df_all[, c("id", "species")]
    this_stack_y$species <- paste0("species_", this_stack_y$species)
    assert_that(!any(!unique(this_stack_y$species) %in% unique(feats_all$feature_name)))
    
    
    source("02_ensemb_xgb_classi_func.R")  # sos_xgb_classi()
    
    
    # actual function call
    returned_thing <- sos_xgb_classi(
        p_feats_all = feats_all,
        p_stack_y = this_stack_y,
        p_tar_var = "species",
        p_train_ids = df_all$id,  # [df_all$dataset == 'train'],  # when p_test_ids is null, we can go ahead and pass in all into train
        # p_test_ids = df_all$id[df_all$dataset == 'test'],  #<-- this will default to NULL, let's make sure that happens
        p_xgb_params = params,
        p_cv_folds = 4,
        p_cv_rounds = 3000,
        p_cv_earlystop = 15,
        p_cheaters = these_cheaters,
        p_read_from_cache = TRUE,
        p_stack_identifier = "01")

    
                    # assertions and inspections
                    assert_that(length(returned_thing) == 1)
                    assert_that(dim(returned_thing[[1]])[[1]] == 450)
                    assert_that(dim(returned_thing[[1]])[[2]] == 3)
                    returned_thing1 <- returned_thing[[1]]
    
                    
    GBLs <- ls()[grepl("^GBL_", ls())]
    rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))
    
                    
# TEST B -------------------------------------------------------------------------
#  Predict on species using all data (no test data specified) and don't read from cache              
    
    
    # pass this in as a parameter itself (the list)
    params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = 3,
                   "eta" = 0.01, "max_depth" = 6, "subsample" = 0.7, "colsample_bytree" = 0.3,
                   # "lambda" = 1.0, # "min_child_weight" = 6, # "gamma" = 10,
                   "alpha" = 1.0, "nthread" = 6)     
    
    
    # cheaters should be based on feature_name within feats_all, NOT within df_all
    these_cheaters <- feats_all$feature_name[grepl("^species_", feats_all$feature_name)] %>% unique()
    
    
    # set up y values for THIS stack, not for overall project
    this_stack_y <- df_all[, c("id", "species")]
    this_stack_y$species <- paste0("species_", this_stack_y$species)
    assert_that(!any(!unique(this_stack_y$species) %in% unique(feats_all$feature_name)))
    
    
    source("02_ensemb_xgb_classi_func.R")  # sos_xgb_classi()
    
    
    # actual function call
    returned_thing <- sos_xgb_classi(
        p_feats_all = feats_all,
        p_stack_y = this_stack_y,
        p_tar_var = "species",
        p_train_ids = df_all$id,  # [df_all$dataset == 'train'],  # when p_test_ids is null, we can go ahead and pass in all into train
        # p_test_ids = df_all$id[df_all$dataset == 'test'],  #<-- this will default to NULL, let's make sure that happens
        p_xgb_params = params,
        p_cv_folds = 4,
        p_cv_rounds = 3000,
        p_cv_earlystop = 15,
        p_cheaters = these_cheaters,
        p_read_from_cache = FALSE,
        p_stack_identifier = "01")
    
    
                    # assertions and inspections
                    assert_that(length(returned_thing) == 1)
                    assert_that(dim(returned_thing[[1]])[[1]] == 450)
                    assert_that(dim(returned_thing[[1]])[[2]] == 3)
                    returned_thing1 <- returned_thing[[1]]            
                    
    
    GBLs <- ls()[grepl("^GBL_", ls())]
    rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))
                       
                    
    
# TEST C -------------------------------------------------------------------------
# Predict on species using train data for train, test data for test, read from cache             


    # pass this in as a parameter itself (the list)
    params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = 3,
                   "eta" = 0.01, "max_depth" = 6, "subsample" = 0.7, "colsample_bytree" = 0.3,
                   # "lambda" = 1.0, # "min_child_weight" = 6, # "gamma" = 10,
                   "alpha" = 1.0, "nthread" = 6)     
    
    
    # cheaters should be based on feature_name within feats_all, NOT within df_all
    these_cheaters <- feats_all$feature_name[grepl("^species_", feats_all$feature_name)] %>% unique()
    
    
    # set up y values for THIS stack, not for overall project
    this_stack_y <- df_all[, c("id", "species")]
    this_stack_y$species <- paste0("species_", this_stack_y$species)
    assert_that(!any(!unique(this_stack_y$species) %in% unique(feats_all$feature_name)))
    
    
    source("02_ensemb_xgb_classi_func.R")  # sos_xgb_classi()
    
    
    # actual function call
    returned_thing <- sos_xgb_classi(
        p_feats_all = feats_all,
        p_stack_y = this_stack_y,
        p_tar_var = "species",
        p_train_ids = df_all$id[df_all$dataset == 'train'], # when p_test_ids is null, we can go ahead and pass in all into train
        p_test_ids = df_all$id[df_all$dataset == 'test'],   #<-- this will default to NULL, let's make sure that happens
        p_xgb_params = params,
        p_cv_folds = 4,
        p_cv_rounds = 3000,
        p_cv_earlystop = 15,
        p_cheaters = these_cheaters,
        p_read_from_cache = TRUE,
        p_stack_identifier = "01")
    
    
                    # assertions and inspections
                    assert_that(length(returned_thing) == 2)
                    assert_that(dim(returned_thing[[1]])[[1]] == 315)
                    assert_that(dim(returned_thing[[1]])[[2]] == 3)
                    returned_thing1 <- returned_thing[[1]]     
                    returned_thing2 <- returned_thing[[2]]
    
    
    GBLs <- ls()[grepl("^GBL_", ls())]
    rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))
    
    

# TEST FUNC (pwid_bin) - binary category of petal width -----------------


    # pass this in as a parameter itself (the list)
    params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = 3,
                   "eta" = 0.01, "max_depth" = 6, "subsample" = 0.7, "colsample_bytree" = 0.3,
                   # "lambda" = 1.0, # "min_child_weight" = 6, # "gamma" = 10,
                   "alpha" = 1.0, "nthread" = 6)     
    
    
    # cheaters should be based on feature_name within feats_all, NOT within df_all
    these_cheaters <- feats_all$feature_name[grepl("^pwid_", feats_all$feature_name)] %>% unique()
    these_cheaters2 <- feats_all$feature_name[grepl("^petal_width$", feats_all$feature_name)] %>% unique()
    
    these_cheaters <- unique(c(these_cheaters, these_cheaters2))  
    
    # set up y values for this stack
    this_stack_y <- df_all[, c("id", "pwid_bin")]
    this_stack_y$pwid_bin <- paste0("pwid_bin_", this_stack_y$pwid_bin)
    assert_that(!any(!unique(this_stack_y$pwid_bin) %in% unique(feats_all$feature_name)))
    
    
    # actual function call
    returned_thing <- sos_xgb_classi(
        p_feats_all = feats_all,
        p_stack_y = this_stack_y,
        p_tar_var = "pwid_bin",
        p_train_ids = df_all$id[df_all$dataset == 'train'],
        p_test_ids = df_all$id[df_all$dataset == 'test'],
        p_xgb_params = params,
        p_cheaters = these_cheaters,
        p_stack_identifier = "01")




