rm(list=ls(all=TRUE))
gc()

pkgs <- c("caret", "data.table", "xgboost", "caTools")
# install.packages(pkgs)

# Load required libraries
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
basePath <- "F:/Education/Kaggle/Homesite/model/xgboost_cv/ensemble/ensemble_4"
setwd(basePath)

tmpData <- fread(paste0("cv_pred_4_", 5, ".csv"))
cv_pred <- tmpData[, "QuoteConversion_Flag", with=FALSE]
for(i in 5:8) {
  tmpData <- fread(paste0("cv_pred_4_", i, ".csv"))
  tmpData[, QuoteConversion_Flag:=NULL]
  cv_pred <- cbind(cv_pred, tmpData)
}
write.csv(cv_pred, "cv_pred_4.csv", row.names=FALSE)

tmpData <- fread(paste0("val_pred_4_", 5, ".csv"))
val_pred <- tmpData[, "QuoteConversion_Flag", with=FALSE]
for(i in 5:8) {
  tmpData <- fread(paste0("val_pred_4_", i, ".csv"))
  tmpData[, QuoteConversion_Flag:=NULL]
  val_pred <- cbind(val_pred, tmpData)
}
write.csv(val_pred, "val_pred_4.csv", row.names=FALSE)
