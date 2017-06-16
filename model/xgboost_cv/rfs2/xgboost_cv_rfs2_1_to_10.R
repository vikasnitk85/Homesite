#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

# Version <- "13"

.libPaths("E:/R-Packages")

# Link Dropbox
# library(RStudioAMI)
# linkDropbox()
pkgs <- c("caret", "data.table", "xgboost", "caTools", "doSNOW", "tcltk")
# install.packages(pkgs)

# Load required libraries
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
# basePath <- "/home/rstudio/Dropbox/Public/Homesite/"
# basePath <- "F:/Education/Kaggle/Homesite/"
basePath <- "D:/Vikas_Agrawal/Education/Kaggle/Homesite/"
setwd(basePath)

# Source New Script of xgb.cv
source("model/xgboost_cv/xgb.cv1.R")

#-------------------------------------------------------------------------------
# Load data
#-------------------------------------------------------------------------------
inputData <- fread("input/train.csv")
inputData[, Original_Quote_Date:=as.Date(Original_Quote_Date, "%Y-%m-%d")]
inputData[, year:=as.numeric(format(Original_Quote_Date, "%y"))]
inputData[, month:=as.numeric(format(Original_Quote_Date, "%m"))]
inputData[, day:=as.numeric(format(Original_Quote_Date, "%d"))]
inputData[, c("QuoteNumber", "Original_Quote_Date"):=NULL]
inputData[, Field10:=as.numeric(gsub(",", "", Field10))]

for(f in names(inputData)) {
  if(class(inputData[[f]]) == "character") {
    inputData[[f]] <- as.numeric(as.factor(inputData[[f]]))
  }
}

# Replace missing values by -1
inputData[is.na(PropertyField29), PropertyField29:=-1]
inputData[is.na(PersonalField84), PersonalField84:=-1]

#-------------------------------------------------------------------------------
# Drop variables with one or zero unique values (after removing missing values i.e. -1)
#-------------------------------------------------------------------------------
dropVars <- c("GeographicField5A", "GeographicField14A", "GeographicField18A", "GeographicField21A",
  "GeographicField22A", "GeographicField23A", "GeographicField56A", "GeographicField60A",
  "GeographicField61A", "GeographicField62A", "PropertyField2A", "PropertyField6", "PropertyField11A",
  "GeographicField10A")
inputData[, c(dropVars):=NULL]

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
# Create index for cross validation
#-------------------------------------------------------------------------------
# set.seed(102)
# index <- createFolds(train_df[, outcome_name], k = 3)

# Define list of parameters
param0 <- list(booster = "gbtree"
               , silent = 0
               , eta = 0.2
               , gamma = 0
               , max_depth = 5
               , min_child_weight = 10
               , subsample = 1
               , colsample_bytree = 1
               , objective = "binary:logistic"
               , eval_metric = "auc"
               )

#-------------------------------------------------------------------------------
# Recursive Feature Selection
#-------------------------------------------------------------------------------
# Selected Vars
selVars <- c("PropertyField37")

for(Version in 1:10) {
  cat("=========================================================\n")
  procStartTime <- Sys.time()
  feature_names <- setdiff(feature_names, selVars)
  cat("Version: ", Version, " Features: ", length(feature_names), "\n")

  rfs2_outputs <- data.table(SelectedVars=feature_names)
  rfs2_outputs[, N:=1:.N]

  cl <- makeCluster(3, type="SOCK")
  registerDoSNOW(cl)

  index <- createFolds(train_df[, outcome_name], k = 3)
  final_outputs <- foreach(i=1:nrow(rfs2_outputs), .inorder=FALSE, .packages=c("xgboost", "data.table", "caTools", "tcltk")) %dopar% {
    if(!exists("pb")) pb <- tkProgressBar("Variables Completed", min=1, max=nrow(rfs2_outputs))
    setTkProgressBar(pb, i)

    startTime <- Sys.time()
    new_features <- c(selVars, rfs2_outputs[i, SelectedVars])
    dtrain <- xgb.DMatrix(data=data.matrix(train_df[, new_features]), label=train_df[, outcome_name])
    dval <- xgb.DMatrix(data=data.matrix(validate_df[, new_features]))

    #-------------------------------------------------------------------------------
    # Fit cross validation model using new_script
    #-------------------------------------------------------------------------------
    set.seed(1234)
    system.time(xgb_cv_new <- xgb.cv1(params=param0, data=dtrain, nrounds=200,
      metrics=list("auc"), folds=index, verbose=FALSE, prediction=TRUE))
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

    rfs2_outputs[i, test_auc_mean:=xgb_cv_perf[tree==new_nrounds, test_auc_mean]]
    rfs2_outputs[i, test_auc_std:=xgb_cv_perf[tree==new_nrounds, test_auc_std]]

    #-------------------------------------------------------------------------------
    # Fit final models
    #-------------------------------------------------------------------------------
    set.seed(1234)
    system.time(xgb_fit0 <- xgb.train(params=param0, data=dtrain, nrounds=new_nrounds)) # 14 Seconds
    tmpPred <- predict(object=xgb_fit0, newdata=dval, ntreelimit=new_nrounds)
    auc <- colAUC(X=tmpPred, y=validate_df[, outcome_name])

    endTime <- Sys.time()
    timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
    rfs2_outputs[i, best_nrounds:=new_nrounds]
    rfs2_outputs[i, val_auc:=as.numeric(auc)]
    rfs2_outputs[i, time_secs:=timeTaken]
    out <- rfs2_outputs[i, ]
    out
  }
  stopCluster(cl)
  procEndTime <- Sys.time()
  procTimeTaken <- difftime(procEndTime, procStartTime, units="secs")
  cat("Time Taken: ", round(procTimeTaken, 0), "\n")

  rfs2_outputs <- rbindlist(final_outputs)
  setorder(rfs2_outputs, N)
  # summary(rfs2_outputs)

  rfs2_outputs[, Avg_AUC:=(test_auc_mean+val_auc)/2]
  rfs2_outputs[, Max_Diff:=max(Avg_AUC)-Avg_AUC]

  missInfo <- fread("model/xgboost_cv/Missing_Value_Analysis.csv")
  rfs2_outputs <- merge(rfs2_outputs, missInfo, by.x="SelectedVars", by.y="Var")
  setnames(rfs2_outputs, "NegativeCount", "MissingValues")
  setorder(rfs2_outputs, test_auc_mean)
  print(head(data.frame(rfs2_outputs)))
  write.csv(rfs2_outputs, paste0("model/xgboost_cv/rfs2/xgboost_cv_rfs2_", Version, ".csv"), row.names=FALSE)

  selVars <- c(selVars, rfs2_outputs[1, SelectedVars])
  cat("Selected Vars: ", paste(selVars, collapse=", "), "\n")
  cat("\n")
}
