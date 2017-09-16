

# xgb sweeper function

#' despite the fact that this script is within this repo, I want this to be extremely stand-alone.


library(xgboost)
library(dplyr)


# mix up the data
myiris <- iris[sample(1:nrow(iris), nrow(iris)), ]
myiris$id <- paste0("id", sprintf("%03.0f", 1:nrow(myiris)))




set.seed(1)
train_indx <- sample(1:nrow(myiris), ceiling(0.7 * nrow(myiris)))


sprintf("")

sprintf("%f", pi)
sprintf("%.3f", pi)
sprintf("%1.0f", pi)
sprintf("%5.1f", pi)
sprintf("%05.1f", pi)
sprintf("%+f", pi)
sprintf("% f", pi)
sprintf("%-10f", pi) # left justified
sprintf("%e", pi)
sprintf("%E", pi)
sprintf("%g", pi)
sprintf("%g",   1e6 * pi) # -> exponential
sprintf("%.9g", 1e6 * pi) # -> "fixed"
sprintf("%G", 1e-6 * pi)