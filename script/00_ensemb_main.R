

# main.R for ensemb_small project

# goal, combine the logic and thought process of these repos
    # https://github.com/Keiku/kaggle-airbnb-recruiting-new-user-bookings  # <-- master of xgboost sparse stacking
    # https://github.com/owenzhang/Kaggle-AmazonChallenge2013              # <-- master of the glmnet formula interactions
    #  ^ Need to figure out his "__final_utils.R" script, really weird functions in there

# and this kaggle script (imputation, skewness, collinearity, blending)
    # https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling


# into my own library of usable functions for any data science project


# Investigate catboost when it is more mature (currently limited to certain Vis Studio compiler)
    # https://www.kaggle.com/jmbull/boost-starter-0-0648x-lb  
    # https://techcrunch.com/2017/07/18/yandex-open-sources-catboost-a-gradient-boosting-machine-learning-librar/ 
    # https://catboost.yandex/#benchmark 
    # https://github.com/catboost/catboost
    # this would mean I don't have to do any of the preprocess work... 


# STEPS - only focus on the very next step which needs to be accomplished:
#' 1) build the regression version of the stage1_multi_func function
#' 2) work on functionizing the preprocessing steps for this pipeline
#' 3) can catboost work on sparse matrix data structures in R? 
#'    - If so, drop it in in-place and test accuracy
#'    - If not, drop it in with the non-numeric, non-sparse data and test




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




# source in the stackers - each of these will read/write to/from disk
# also, each of these will have uniform inputs made available to them
# each will have a standardized "config" at the top of the file 
# for that individual stacker
# each of these will clean up global env after completion and restore
# global env to what it looked like after 01_ensemb_preproc.R ran.
# each will conduct garbage collection to free up memory.






# notes on train/test split for low level stackers:
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




