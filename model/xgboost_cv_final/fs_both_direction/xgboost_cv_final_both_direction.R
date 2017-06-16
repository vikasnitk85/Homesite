#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

# Link Dropbox
# library(RStudioAMI)
# linkDropbox()
pkgs <- c("caret", "data.table", "xgboost", "caTools", "doSNOW", "tcltk")
# install.packages(pkgs)

# Load required libraries
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
# basePath <- "/home/rstudio/Dropbox/Public/Homesite/"
basePath <- "F:/Education/Kaggle/Homesite/"
setwd(basePath)

# Source New Script of xgb.cv
source("model/xgboost_cv_final/xgb.cv1.R")

#-------------------------------------------------------------------------------
# Data preparation
#-------------------------------------------------------------------------------
# Load data
inputData <- fread("input/train.csv")
inputData[, WeekDay:=as.POSIXlt(as.Date(Original_Quote_Date, "%Y-%m-%d"))$wday]
inputData[, c("QuoteNumber", "Original_Quote_Date"):=NULL]
inputData[, Field10:=as.numeric(gsub(",", "", Field10))]

# Convert categorical variables to numeric
for(f in names(inputData)) {
  if(class(inputData[[f]]) == "character") {
    inputData[[f]] <- as.numeric(as.factor(inputData[[f]]))
  }
}

# Replace missing values by -1
inputData[is.na(PropertyField29), PropertyField29:=-1]
inputData[is.na(PersonalField84), PersonalField84:=-1]

# Convert integer variables to numeric
for(f in names(inputData)) {
  if(class(inputData[[f]]) == "integer") {
    inputData[[f]] <- as.numeric(inputData[[f]])
  }
}

# Drop variables with one or zero unique values (after removing missing values i.e. -1)
dropVars <- c("GeographicField5A", "GeographicField14A", "GeographicField18A", "GeographicField21A",
  "GeographicField22A", "GeographicField23A", "GeographicField56A", "GeographicField60A",
  "GeographicField61A", "GeographicField62A", "PropertyField2A", "PropertyField6", "PropertyField11A",
  "GeographicField10A")
inputData[, c(dropVars):=NULL]

# Count of -1 and zeros by observations
inputData[, count_less_0:=apply(inputData[, -1, with=FALSE], 1, function(x) sum(x < 0))]
inputData[, count_0:=apply(inputData[, -1, with=FALSE], 1, function(x) sum(x == 0))]

#-------------------------------------------------------------------------------
# List of features
#-------------------------------------------------------------------------------
outcome_name <- "QuoteConversion_Flag"
feature_names <- setdiff(names(inputData), outcome_name)

#-------------------------------------------------------------------------------
# Prepare framework for model development
#-------------------------------------------------------------------------------
# Define list of parameters
nrounds <- 200
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
# Feature selection
#-------------------------------------------------------------------------------
# List of initially selected variables
selVars <- c("PropertyField37", "PersonalField10A", "SalesField5")

for(Version in 1:35) {
  procStartTime <- Sys.time()
  cat("=========================================================\n")

  # List of remaining variables
  new_features <- setdiff(feature_names, selVars)

  # Create index for cross-validation folds
  Seed <- sample.int(4096, size=1) - 1
  set.seed(Seed)
  index <- createFolds(inputData[, outcome_name], k = 3)

  # Initiate clusters
  cl <- makeCluster(3, type="SOCK")
  registerDoSNOW(cl)

  #-------------------------------------------------------------------------------
  # Forward selection
  #-------------------------------------------------------------------------------
  cat("Version: ", Version, " Features: ", length(new_features), " Index Seed: ", Seed, "\n")

  # Initiate outputs
  fs_start_time <- Sys.time()
  tmpOut <- data.table(SelectedVars=new_features)
  tmpOut[, N:=1:.N]

  forward_out <- foreach(i=1:nrow(tmpOut), .inorder=FALSE, .packages=c("xgboost", "data.table", "caTools", "tcltk")) %dopar% {
    if(!exists("pb")) pb <- tkProgressBar("Variables Completed", min=1, max=nrow(tmpOut))
    setTkProgressBar(pb, i)

    startTime <- Sys.time()
    tmp_features <- c(selVars, tmpOut[i, SelectedVars])
    dtrain <- xgb.DMatrix(data=data.matrix(inputData[, tmp_features]), label=inputData[, outcome_name])

    # Fit cross validation model using modified xgb.cv
    set.seed(1234)
    system.time(xgb_cv_new <- xgb.cv1(params=param0, data=dtrain, nrounds=nrounds,
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

    tmpOut[i, test_auc_mean:=xgb_cv_perf[tree==new_nrounds, test_auc_mean]]
    tmpOut[i, test_auc_std:=xgb_cv_perf[tree==new_nrounds, test_auc_std]]

    endTime <- Sys.time()
    timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
    tmpOut[i, best_nrounds:=new_nrounds]
    tmpOut[i, time_secs:=timeTaken]
    out <- tmpOut[i, ]
    out
  }
  fs_end_time <- Sys.time()
  fs_time <- difftime(fs_end_time, fs_start_time)
  forward_out <- rbindlist(forward_out)
  forward_out[, TimeTaken:=fs_time]
  setorder(forward_out, -test_auc_mean)
  write.csv(forward_out, paste0("model/xgboost_cv_final/fs_both_direction/xgboost_cv_final_fs_", Version, ".csv"), row.names=FALSE)
  print(head(data.frame(forward_out)))

  #-------------------------------------------------------------------------------
  # Backward elimination
  #-------------------------------------------------------------------------------
  selVars <- c(selVars, forward_out[1, SelectedVars])
  tmpVars <- c(selVars, "")

  # Initiate outputs
  bs_start_time <- Sys.time()
  tmpOut <- data.table(DroppedVar=tmpVars)
  tmpOut[, N:=1:.N]

  backward_out <- foreach(i=1:nrow(tmpOut), .inorder=FALSE, .packages=c("xgboost", "data.table", "caTools", "tcltk")) %dopar% {
    tmp_features <- setdiff(selVars, tmpVars[i])

    startTime <- Sys.time()
    dtrain <- xgb.DMatrix(data=data.matrix(inputData[, tmp_features]), label=inputData[, outcome_name])

    # Fit cross validation model using modified xgb.cv
    set.seed(1234)
    system.time(xgb_cv_new <- xgb.cv1(params=param0, data=dtrain, nrounds=nrounds,
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

    tmpOut[i, test_auc_mean:=xgb_cv_perf[tree==new_nrounds, test_auc_mean]]
    tmpOut[i, test_auc_std:=xgb_cv_perf[tree==new_nrounds, test_auc_std]]

    endTime <- Sys.time()
    timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
    tmpOut[i, best_nrounds:=new_nrounds]
    tmpOut[i, time_secs:=timeTaken]
    out <- tmpOut[i, ]
    out
  }
  bs_end_time <- Sys.time()
  bs_time <- difftime(bs_end_time, bs_start_time)
  backward_out <- rbindlist(backward_out)
  backward_out[, TimeTaken:=bs_time]
  setorder(backward_out, -test_auc_mean)
  write.csv(backward_out, paste0("model/xgboost_cv_final/fs_both_direction/xgboost_cv_final_bs_", Version, ".csv"), row.names=FALSE)
  print(head(data.frame(backward_out)))

  #-------------------------------------------------------------------------------
  # List of final selected variables
  #-------------------------------------------------------------------------------
  selVars <- setdiff(selVars, backward_out[1, DroppedVar])

  stopCluster(cl)
  procEndTime <- Sys.time()
  procTimeTaken <- difftime(procEndTime, procStartTime, units="secs")
  cat("Time Taken: ", round(procTimeTaken, 0), "\n")
  cat("Selected Vars: ", paste(selVars, collapse=", "), "\n")
}
