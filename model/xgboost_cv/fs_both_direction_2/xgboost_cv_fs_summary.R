rm(list=ls(all=TRUE))
options(scipen=999)
setwd("D:/Vikas_Agrawal/Education/Kaggle/Homesite/model/xgboost_cv/fs_both_direction_2/")
library(data.table)
completed <- 22

#------------------------------------------------------------------------------
# Forward Selection Summary
#------------------------------------------------------------------------------
finalOut <- NULL
for(Ver in 1:completed) {
  tmpData <- fread(paste0("xgboost_cv_fs_2_", Ver, ".csv"))
  tmpData[, Version:=Ver]
  finalOut <- rbind(finalOut, tmpData)
}

setorder(finalOut, Version, -test_auc_mean)
bestOut <- finalOut[, .SD[1], by="Version"]
bestOut[, Improvement_test:=test_auc_mean-shift(test_auc_mean, 1)]
bestOut[, Improvement_val:=val_auc-shift(val_auc, 1)]
setnames(bestOut, "SelectedVars", "AddedVariable")
bestOut[, N:=NULL]
bestOut[, time_secs:=NULL]
bestOut[, test_auc_std:=NULL]
bestOut[, Max_Diff:=NULL]
setcolorder(bestOut, c("Version", "AddedVariable", "test_auc_mean", "Improvement_test", "val_auc",
  "Improvement_val", "TimeTaken"))
bestOut
write.csv(bestOut, "xgboost_cv_fs_2_summary.csv", row.names=FALSE)

#------------------------------------------------------------------------------
# Backward Selection Summary
#------------------------------------------------------------------------------
finalOut <- NULL
for(Ver in 1:completed) {
  tmpData <- fread(paste0("xgboost_cv_bs_2_", Ver, ".csv"))
  tmpData[, Version:=Ver]
  finalOut <- rbind(finalOut, tmpData)
}

setorder(finalOut, Version, -test_auc_mean)
bestOut <- finalOut[, .SD[1], by="Version"]
bestOut[, Improvement_test:=test_auc_mean-shift(test_auc_mean, 1)]
bestOut[, Improvement_val:=val_auc-shift(val_auc, 1)]
bestOut[, N:=NULL]
bestOut[, time_secs:=NULL]
bestOut[, test_auc_std:=NULL]
bestOut[, Max_Diff:=NULL]
setcolorder(bestOut, c("Version", "DroppedVar", "test_auc_mean", "Improvement_test", "val_auc",
  "Improvement_val", "TimeTaken"))
bestOut
write.csv(bestOut, "xgboost_cv_bs_2_summary.csv", row.names=FALSE)
