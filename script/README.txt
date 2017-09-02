
Ensemble Stacker

file structure and naming conventions:

00_ensemb_main.R - start here, this is what will run the whole project
01_ensemb_preproc.R - this is called by 00_ensemb_main.R and will preprocess the iris data

zz_utils_small.R - called by 00_ensemb_main.R as well as other scripts while testing


NOTE ON PREFIXES:
02 prefix - reserved for all of the stage1 stackers. these pull in data from 01_ensemb_preproc.R output and prepare output for the 03 prefix files

03 prefix - reserved for all of the stage2 stackers. these will pull in output from 02 prefix output and train on the overall target variable for the project as a whole (assuming only two layers to the stack). 


02_ensemb_multi.R - this is the current best implementation for a multi-label (more than 2) classification stage 1 stacker. Think of the "species" column in the iris dataset.

02_ensemb_multi_tpot.ipynb - same as 02_ensemb_multi.R, but I want this one to try and utilize a R/Python combined pipeline and it will be used primarily to determine the best type of model for the stacker to use. This won't be part of the main pipeline, but instead it will be used to determine which model to embed in the 02_ensemb_multi.R file. This concept generalizes to all of the files that contain the string "tpot" in the name of the file.


TODO:
02_ensemb_regr.R - stage 1 regression stacker
02_ensemb_regr_tpot.ipynb - exploratory, find best model(s) to stack 
02_ensemb_bin.R - stage 1 binary classification stacker
02_ensemb_bin_tpot.ipynb - exploratory, find best model(s) to stack

(I want all of the 02 combinations above to also be built into 03 combinations)

REMEMBER: 
This is setting up the STRUCTURE of a future project. Ideally, you would have many of each of these files. Each .R file would be it's own stage1 or stage2 stacker and the title would indicate what it is using as a target (and possible as a training set if the same target is being put up against multiple forms of training data).


SCOPE:
all 03 files should correspond with what the overall project is attempting to predict. we are assuming a simple two-layer stacker for this first iteration.



