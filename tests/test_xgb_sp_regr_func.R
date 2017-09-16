
# test_xgb_sp_regr_func

# main setup ---------------------

    # set working directory (based on which machine I'm currently using)
    if(Sys.info()[["sysname"]] == "Windows" & Sys.info()[["user"]] == "Taylor") {
        setwd("C:/Users/Taylor/Documents/Nerd/github_repos/stacks_on_stacks/script")
    } else {
        setwd("C:/Users/tvananne/Documents/personal/github/stacks_on_stacks/script")
    }
    getwd()
    
    # source in utilities script
    source('zz_ensemb_utils.R')  # <-- this should not remove items from environment
    
    if(TRUE) {
        source("01_ensemb_sparse_preproc.R")  # <-- standardized rules for preproc
        # complex datasets will have several preproc / first stage feature generation files
    }
    
    
    
# TEST A --------------------------
    
    # pass this in as a parameter itself (the list)
    params <- list("objective"="reg:linear", "eval_metric"="rmse", 
                   "eta"=0.01, "max_depth"=6, "colsample_bytree" = 0.8,
                   "alpha" = 1.0, "nthread" = 6)     
    
    
    this_stack_y <- df_all[, c("id", "petal_width")]
    
    
    these_cheaters <- unique(feats_all$feature_name[grepl("^pwid_bin_", feats_all$feature_name)])
    these_cheaters

    source("02_ensemb_xgb_regr_func.R")
    
    
    return_thing <- sos_xgb_regr(
        p_feats_all = feats_all,             
        p_stack_y = this_stack_y,             
        p_tar_var = "petal_width",                
        p_train_ids <- df_all$id,   #  [df_all$dataset == 'train']
        # p_test_ids  <- df_all$id[df_all$dataset == 'test']  
        p_xgb_params = params,
        p_cv_folds=4,
        p_cv_rounds=3000,
        p_cv_earlystop=15,
        p_cheaters = these_cheaters,               # <-- c("") -- vector
        p_read_from_cache=FALSE,  
        p_stack_identifier="04"
        ) 
    

            # return thing
            ret1 <- return_thing[[1]]
    
    
    
    
            
    