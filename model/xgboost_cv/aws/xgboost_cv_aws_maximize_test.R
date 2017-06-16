#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

# Link Dropbox
# library(RStudioAMI)
# linkDropbox()
pkgs <- c("caret", "data.table", "xgboost", "caTools")
# install.packages(pkgs)

# Load required libraries
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
basePath <- "/home/rstudio/Dropbox/Public/Homesite/"
setwd(basePath)

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
# train_df[, outcome_name] <- ifelse(train_df[, outcome_name]==0, "N", "Y")
# train_df[, outcome_name] <- as.factor(train_df[, outcome_name])
dim(train_df)
validate_df <- data.frame(inputData[random_splits >=.5, ])
dim(validate_df)

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
paramList <- expand.grid(eta=c(0.02), max_depth=c(5:8), min_child_weight=c(10),
  subsample=c(0.9), colsample_bytree=c(0.6))
paramList <- data.table(paramList)
setorder(paramList, eta, max_depth, min_child_weight, subsample, colsample_bytree)
paramList[max_depth==5, nrounds:=2000]
paramList[max_depth==6, nrounds:=1700]
paramList[max_depth==7, nrounds:=1400]
paramList[max_depth==8, nrounds:=1100]

ens_pred <- data.frame(QuoteConversion_Flag=validate_df[, outcome_name])
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
  # Fit cross validation models
  #-------------------------------------------------------------------------------
  set.seed(1234)
  system.time(xgb_cv0 <- xgb.cv(params=param0, data=dtrain, nrounds=paramList[i, nrounds], 
    metrics=list("auc"), folds=index, verbose=FALSE, maximize=FALSE))
  setnames(xgb_cv0, gsub("[.]", "_", names(xgb_cv0)))
  xgb_cv0[, test_auc_mean_sd:=test_auc_mean-test_auc_std]
  xgb_cv0[, tree:=1:.N]
  setorder(xgb_cv0, -test_auc_mean)
  tree_auc_mean <- xgb_cv0[1, tree]
  setorder(xgb_cv0, -test_auc_mean_sd)
  tree_auc_mean_sd <- xgb_cv0[1, tree]

  #-------------------------------------------------------------------------------
  # Fit final models
  #-------------------------------------------------------------------------------
  new_nrounds <- max(tree_auc_mean_sd, tree_auc_mean) + 10
  set.seed(1234)
  system.time(xgb_fit0 <- xgb.train(params=param0, data=dtrain, nrounds=new_nrounds, maximize=FALSE)) # 207.14 seconds

  #-------------------------------------------------------------------------------
  # Predict validation dataset using best nrounds by test_auc_mean
  #-------------------------------------------------------------------------------
  pred_auc_mean <- predict(object=xgb_fit0, newdata=dval, ntreelimit=tree_auc_mean)
  ens_pred <- cbind(ens_pred, pred_auc_mean)

  #-------------------------------------------------------------------------------
  # Predict validation dataset using best nrounds by test_auc_mean_sd
  #-------------------------------------------------------------------------------
  pred_auc_mean_sd <- predict(object=xgb_fit0, newdata=dval, ntreelimit=tree_auc_mean_sd)

  #-------------------------------------------------------------------------------
  # Calculate ROC AUC
  #-------------------------------------------------------------------------------
  pred_val <- data.frame(pred_auc_mean, pred_auc_mean_sd)
  auc <- colAUC(X=pred_val, y=validate_df[, outcome_name])

  #-------------------------------------------------------------------------------
  # Save output details
  #-------------------------------------------------------------------------------
  endTime <- Sys.time()
  timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
  paramList[i, nrounds_auc_mean:=tree_auc_mean]
  paramList[i, test_auc_mean:=xgb_cv0[tree==tree_auc_mean, test_auc_mean]]
  paramList[i, nrounds_auc_mean_sd:=tree_auc_mean_sd]
  paramList[i, test_auc_mean_sd:=xgb_cv0[tree==tree_auc_mean_sd, test_auc_mean_sd]]
  paramList[i, val_auc_mean:=auc[1]]
  paramList[i, val_auc_mean_sd:=auc[2]]
  paramList[i, time_secs:=timeTaken]
  print(data.frame(paramList)[i, ])
}
names(ens_pred) <- c("QuoteConversion_Flag", paste0("eta_", 5:8))
ens_pred <- data.frame(ens_pred)



Weight <- paramList[, test_auc_mean]/sum(paramList[, test_auc_mean])
ens_pred$ens1 <- rowMeans(ens_pred[, paste0("eta_", c(5, 6, 7, 8))])
ens_pred$ens2 <- rowMeans(ens_pred[, paste0("eta_", c(5, 6, 7))])
ens_pred$ens3 <- rowMeans(ens_pred[, paste0("eta_", c(5, 6, 8))])
ens_pred$ens4 <- rowMeans(ens_pred[, paste0("eta_", c(5, 7, 8))])
ens_pred$ens5 <- rowMeans(ens_pred[, paste0("eta_", c(6, 7, 8))])
ens_pred$ens6 <- rowMeans(ens_pred[, paste0("eta_", c(5, 6))])
ens_pred$ens7 <- rowMeans(ens_pred[, paste0("eta_", c(5, 7))])
ens_pred$ens8 <- rowMeans(ens_pred[, paste0("eta_", c(5, 8))])
ens_pred$ens9 <- rowMeans(ens_pred[, paste0("eta_", c(6, 7))])
ens_pred$ens10 <- rowMeans(ens_pred[, paste0("eta_", c(6, 8))])
ens_pred$ens11 <- rowMeans(ens_pred[, paste0("eta_", c(7, 8))])

auc <- colAUC(X=ens_pred[, -1], y=ens_pred[, 1])
auc

            # eta_5     eta_6     eta_7     eta_8    ens1      ens2    ens3      ens4      ens5      ens6      ens7
# 0 vs. 1 0.9655105 0.9655369 0.9654955 0.9654093 0.96565 0.9656381 0.96564 0.9656365 0.9656112 0.9655972 0.9656235
             # ens8      ens9     ens10    ens11
# 0 vs. 1 0.9656174 0.9655998 0.9655921 0.965543

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

Weights <- greedOptAUC(X=as.matrix(ens_pred[, paste0("eta_", 5:8)]), Y=ens_pred[, 1])
Weights <- Weights/sum(Weights)
ens_pred$ens12 <- as.matrix(ens_pred[, paste0("eta_", 5:8)]) %*% Weights
auc <- colAUC(X=ens_pred[, -1], y=ens_pred[, 1])
auc

