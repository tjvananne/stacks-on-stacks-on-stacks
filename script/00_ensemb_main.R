

# main.R for ensemb_small project
# much of this project structure is based on this awesome
# second-place submission in the airbnb kaggle competition:
# https://github.com/Keiku/kaggle-airbnb-recruiting-new-user-bookings



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
    source("01_ensemb_preproc.R")  # <-- standardized rules for preproc
    # complex datasets will have several preproc / first stage feature generation files
}




# source in the stackers - each of these will read/write to/from disk
# also, each of these will have uniform inputs made available to them
# each will have a standardized "config" at the top of the file 
# for that individual stacker
# each of these will clean up global env after completion and restore
# global env to what it looked like after 01_ensemb_preproc.R ran.
# each will conduct garbage collection to free up memory.
source("02_ensemb_stage1_multi.R")





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




