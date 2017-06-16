#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

procStartTime <- Sys.time()
Version <- "4"

# Link Dropbox
# library(RStudioAMI)
# linkDropbox()
pkgs <- c("caret", "data.table", "xgboost", "caTools")
# install.packages(pkgs)

# Load required libraries
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
# basePath <- "/home/rstudio/Dropbox/Public/Homesite/"
basePath <- "F:/Education/Kaggle/Homesite/"
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
paramList <- expand.grid(eta=c(0.02), max_depth=c(5:8), min_child_weight=c(5, 10, 15),
  subsample=seq(0.6, 1.0, 0.1), colsample_bytree=seq(0.6, 1.0, 0.1))
paramList <- data.table(paramList)
setorder(paramList, eta, max_depth, min_child_weight, subsample, colsample_bytree)
paramList[max_depth==5, nrounds:=2000]
paramList[max_depth==6, nrounds:=1500]
paramList[max_depth==7, nrounds:=1250]
paramList[max_depth==8, nrounds:=1100]
paramList[, N:=1:.N]

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

paramList <- fread(paste0("model/xgboost_cv/ensemble/ensemble_", Version, "/xgboost_cv_ensemble_", Version, ".csv"))
cv_pred <- fread(paste0("model/xgboost_cv/ensemble/ensemble_", Version, "/cv_pred_", Version, ".csv"))
val_pred <- fread(paste0("model/xgboost_cv/ensemble/ensemble_", Version, "/val_pred_", Version, ".csv"))

cv_pred <- data.frame(cv_pred)
cv_auc <- (colAUC(X=cv_pred[index$Fold1, -1], y=cv_pred[index$Fold1, 1]) +
           colAUC(X=cv_pred[index$Fold2, -1], y=cv_pred[index$Fold2, 1]) +
           colAUC(X=cv_pred[index$Fold3, -1], y=cv_pred[index$Fold3, 1]))/3
summary(as.numeric(cv_auc))
val_pred <- data.frame(val_pred)
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
summary(as.numeric(val_auc))

#-------------------------------------------------------------------------------
# Ensemble using xgboost
#-------------------------------------------------------------------------------
param0 <- list(booster = "gbtree"
               , silent = 0
               , eta = 0.02
               , gamma = 0
               , max_depth = 5
               , min_child_weight = 10
               , subsample = 0.9
               , colsample_bytree = 0.6
               , objective = "binary:logistic"
               , eval_metric = "auc"
               )

feature_names <- paste0("N_", paramList[, N])
dtrain <- xgb.DMatrix(data=data.matrix(cv_pred[, feature_names]), label=train_df[, outcome_name])
dval <- xgb.DMatrix(data=data.matrix(val_pred[, feature_names]), label=val_pred[, outcome_name])

set.seed(1234)
index <- createFolds(cv_pred[, outcome_name], k = 5)

set.seed(1234)
system.time(xgb_cv_ens <- xgb.cv1(params=param0, data=dtrain, nrounds=400,
  metrics=list("auc"), folds=index, verbose=TRUE, prediction=TRUE))
xgb_cv_ens <- xgb_cv_ens$dt
setnames(xgb_cv_ens, gsub("[.]", "_", names(xgb_cv_ens)))
xgb_cv_ens[, test_auc_mean_sd:=test_auc_mean-test_auc_std]
xgb_cv_ens[, tree:=1:.N]
setorder(xgb_cv_ens, -test_auc_mean)
tree_auc_mean <- xgb_cv_ens[1, tree]
setorder(xgb_cv_ens, -test_auc_mean_sd)
tree_auc_mean_sd <- xgb_cv_ens[1, tree]
setorder(xgb_cv_ens, tree)

set.seed(1234)
xgb_ens_fit <- xgb.train(params=param0, data=dtrain, nrounds=max(tree_auc_mean, tree_auc_mean_sd))
val_pred$ens1 <- predict(object=xgb_ens_fit, newdata=dval, ntreelimit=max(tree_auc_mean, tree_auc_mean_sd))
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

varImpXGB <- xgb.importance(feature_names = feature_names, model = xgb_ens_fit)

#-------------------------------------------------------------------------------
# Ensemble using cv_auc
#-------------------------------------------------------------------------------
Weights <- as.numeric(cv_auc)/sum(as.numeric(cv_auc))
val_pred$ens2 <- as.matrix(val_pred[, paste0("N_", paramList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Function for finding the optimal weights
#-------------------------------------------------------------------------------
greedOptAUC <- function(X, Y, iter = 100L) {
    N <- ncol(X)
    weights <- rep(0L, N)
    pred <- 0 * X
    sum.weights <- 0L
    stopper <- max(colAUC(X, Y))
    while (sum.weights < iter) {
        sum.weights <- sum.weights + 1L
        pred <- (pred + X) * (1L/sum.weights)
        errors <- colAUC(pred, Y)
        best <- which.max(errors)
        weights[best] <- weights[best] + 1L
        pred <- pred[, best] * sum.weights
    }
    maxtest <- colAUC(X %*% weights, Y)
    if (stopper > maxtest) {
        testresult <- round(maxtest/stopper, 5) * 100
        wstr <- paste0("Optimized weights not better than best model. Ensembled result is ",
            testresult, "%", " of best model AUC. Try more iterations.")
        message(wstr)
    }
    return(weights)
}

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 1 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:1], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens3 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 2 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:2], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens4 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 3 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:3], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens5 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 4 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:4], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens6 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 5 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:5], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens7 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 6 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:6], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens8 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 7 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:7], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens9 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 8 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:8], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens10 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 9 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:9], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens11 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc by max_depth (Top 10 models)
#-------------------------------------------------------------------------------
setorder(paramList, max_depth, -test_auc_mean)
tmpParamList <- paramList[, .SD[1:10], by="max_depth"]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens12 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top variables by xgboost
#-------------------------------------------------------------------------------
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, varImpXGB$Feature[1:10]]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens13 <- as.matrix(val_pred[, varImpXGB$Feature[1:10]]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc

#-------------------------------------------------------------------------------
# Ensemble using top cv_auc models (5 Models)
#-------------------------------------------------------------------------------
setorder(paramList, -test_auc_mean)
tmpParamList <- paramList[1:5, ]
system.time(Weights <- greedOptAUC(X=as.matrix(cv_pred[, paste0("N_", tmpParamList[, N])]), Y=cv_pred[, 1], iter=100))
Weights <- Weights/sum(Weights)
Weights
val_pred$ens14 <- as.matrix(val_pred[, paste0("N_", tmpParamList[, N])]) %*% Weights
val_auc <- colAUC(X=val_pred[, -1], y=val_pred[, 1])
val_auc


procEndTime <- Sys.time()
procTimeTaken <- difftime(procEndTime, procStartTime)
save.image(file=paste0("model/xgboost_cv/ensemble/ensemble_", Version, "/xgboost_cv_ensemble_", Version, ".RData"))
