




# local script config -------------------------
    
    train_size <- 0.7  # between 0.1 - 0.9



# load data ----------------------------
    
    # some of this is obviously iris-specific
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
    
    


# Additional feature generation and data cleansing -----------
    
    # this section will likely require several 01-level scripts
    # this example is simple enough to take just a few lines of code
    
    
    
    # let's create a few more categorical variables so we have more than just 1
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
        # assign(x = paste0("pos_values_", feat), value = as.character(unique(df_feats_[, 2])))
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
    
    
    # remove items ending with underscore in current global env
    rm(list = ls()[grepl("_$", ls())])
    
    
    
    # exposing for other usage:
    df_all <- myiris; head(df_all); dim(df_all)  # wide
    head(feats_all); dim(feats_all)              # long
    head(y); dim(y)
    

    # cache the wide/long data and the answer file 
    saveRDS(df_all, file.path(GBL_cache_path, "df_all.rds"))
    saveRDS(feats_all, file.path(GBL_cache_path, "feats_all.rds"))
    saveRDS(y, file.path(GBL_cache_path, "y.rds"))
    
    
    # mimic passing in just df_all (with tagged train/test field), feats_all (stacked features), y (labels + id)
    GBLs <- ls()[grepl("^GBL_", ls())]
    rm(list = setdiff(ls(), c(GBLs, "df_all", "feats_all", "y")))
    
    