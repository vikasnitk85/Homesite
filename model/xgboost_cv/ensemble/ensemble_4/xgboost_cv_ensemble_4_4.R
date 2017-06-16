#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

procStartTime <- Sys.time()
Version <- "4_4"

# Link Dropbox
# library(RStudioAMI)
# linkDropbox()
pkgs <- c("caret", "data.table", "xgboost", "caTools")
# install.packages(pkgs)

# Load required libraries
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
basePath <- "/home/rstudio/Dropbox/Public/Homesite/"
# basePath <- "F:/Education/Kaggle/Homesite/"
setwd(basePath)

# Source New Script of xgb.cv
source("model/xgboost_cv/ensemble/xgb.cv1.R")

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
inputData <- fread("input/train.csv")
inputData[, Original_Quote_Date:=as.Date(Original_Quote_Date, "%Y-%m-%d")]
inputData[, year:=as.numeric(format(Original_Quote_Date, "%y"))]
inputData[, month:=as.numeric(format(Original_Quote_Date, "%m"))]
inputData[, day:=as.numeric(format(Original_Quote_Date, "%d"))]
inputData[, c("QuoteNumber", "Original_Quote_Date"):=NULL]
inputData[is.na(inputData)] <- 0
inputData[, Field10:=as.numeric(gsub(",", "", Field10))]

for(f in names(inputData)) {
  if(class(inputData[[f]]) == "character") {
    inputData[[f]] <- as.numeric(as.factor(inputData[[f]]))
  }
}

#-------------------------------------------------------------------------------
# Split data into train and test
#-------------------------------------------------------------------------------
outcome_name <- "QuoteConversion_Flag"
feature_names <- setdiff(names(inputData), outcome_name)

set.seed(1234)
random_splits <- runif(nrow(inputData))
train_df <- inputData[random_splits < .5, ]
train_df <- data.frame(train_df)
validate_df <- data.frame(inputData[random_splits >=.5, ])

# Remove constant variables
feature_names <- setdiff(feature_names, names(which(apply(train_df[, feature_names], 2, sd)==0)))

prop.table(table(inputData[, outcome_name, with=FALSE]))
prop.table(table(train_df[, outcome_name]))
prop.table(table(validate_df[, outcome_name]))

#-------------------------------------------------------------------------------
# Prepare train_df and validate_df for xgboost
#-------------------------------------------------------------------------------
set.seed(102)
index <- createFolds(train_df[, outcome_name], k = 3)
dtrain <- xgb.DMatrix(data=data.matrix(train_df[, feature_names]), label=train_df[, outcome_name])
dval <- xgb.DMatrix(data=data.matrix(validate_df[, feature_names]))

#-------------------------------------------------------------------------------
# Search for best set of parameters
#-------------------------------------------------------------------------------
paramList <- expand.grid(eta=c(0.02), max_depth=c(4), min_child_weight=c(5, 10, 15),
  subsample=seq(0.6, 1.0, 0.1), colsample_bytree=seq(0.6, 1.0, 0.1))
paramList <- data.table(paramList)
setorder(paramList, eta, max_depth, min_child_weight, subsample, colsample_bytree)
paramList[max_depth==4, nrounds:=2500]
# paramList[max_depth==5, nrounds:=2000]
# paramList[max_depth==6, nrounds:=700]
# paramList[max_depth==7, nrounds:=550]
# paramList[max_depth==8, nrounds:=400]
paramList[, N:=(1:.N)+300]

val_pred <- data.table(QuoteConversion_Flag=validate_df[, outcome_name])
cv_pred <- data.table(QuoteConversion_Flag=train_df[, outcome_name])

for(i in 1:nrow(paramList)) {
  startTime <- Sys.time()
  param0 <- list(booster = "gbtree"
                 , silent = 0
                 , eta = paramList[i, eta]
                 , gamma = 0
                 , max_depth = paramList[i, max_depth]
                 , min_child_weight = paramList[i, min_child_weight]
                 , subsample = paramList[i, subsample]
                 , colsample_bytree = paramList[i, colsample_bytree]
                 , objective = "binary:logistic"
                 , eval_metric = "auc"
                 )

  #-------------------------------------------------------------------------------
  # Fit cross validation model using new_script
  #-------------------------------------------------------------------------------
  set.seed(1234)
  system.time(xgb_cv_new <- xgb.cv1(params=param0, data=dtrain, nrounds=paramList[i, nrounds],
    metrics=list("auc"), folds=index, verbose=FALSE, prediction=TRUE))
  cv_pred[, paste0("N_", paramList[i, N]):=xgb_cv_new$pred]
  write.csv(cv_pred, paste0("model/xgboost_cv/ensemble/cv_pred_", Version, ".csv"), row.names=FALSE)
  xgb_cv_perf <- xgb_cv_new$dt
  setnames(xgb_cv_perf, gsub("[.]", "_", names(xgb_cv_perf)))
  xgb_cv_perf[, test_auc_mean_sd:=test_auc_mean-test_auc_std]
  xgb_cv_perf[, tree:=1:.N]
  setorder(xgb_cv_perf, -test_auc_mean)
  tree_auc_mean <- xgb_cv_perf[1, tree]
  setorder(xgb_cv_perf, -test_auc_mean_sd)
  tree_auc_mean_sd <- xgb_cv_perf[1, tree]
  setorder(xgb_cv_perf, tree)
  new_nrounds <- max(tree_auc_mean_sd, tree_auc_mean)

  #-------------------------------------------------------------------------------
  # Fit final models
  #-------------------------------------------------------------------------------
  set.seed(1234)
  system.time(xgb_fit0 <- xgb.train(params=param0, data=dtrain, nrounds=new_nrounds)) # 207.14 seconds

  #-------------------------------------------------------------------------------
  # Predict validation dataset
  #-------------------------------------------------------------------------------
  tmpPred <- predict(object=xgb_fit0, newdata=dval, ntreelimit=new_nrounds)
  val_pred[, paste0("N_", paramList[i, N]):=tmpPred]
  write.csv(val_pred, paste0("model/xgboost_cv/ensemble/val_pred_", Version, ".csv"), row.names=FALSE)

  #-------------------------------------------------------------------------------
  # Calculate ROC AUC
  #-------------------------------------------------------------------------------
  auc <- colAUC(X=tmpPred, y=validate_df[, outcome_name])

  #-------------------------------------------------------------------------------
  # Save output details
  #-------------------------------------------------------------------------------
  endTime <- Sys.time()
  timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
  paramList[i, best_nrounds:=new_nrounds]
  paramList[i, test_auc_mean:=xgb_cv_perf[tree==new_nrounds, test_auc_mean]]
  paramList[i, test_auc_std:=xgb_cv_perf[tree==new_nrounds, test_auc_std]]
  paramList[i, val_auc:=as.numeric(auc)]
  paramList[i, time_secs:=timeTaken]
  print(data.frame(paramList)[i, ])
  write.csv(paramList, paste0("model/xgboost_cv/ensemble/xgboost_cv_ensemble_", Version, ".csv"), row.names=FALSE)
}
