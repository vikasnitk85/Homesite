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
basePath <- "C:/Homesite/"
setwd(basePath)

# Source New Script of xgb.cv
source("model/xgboost_cv/xgb.cv1.R")

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
tmpData <- apply(inputData[, -1, with=FALSE], 1, function(x) sum(x < 0))
inputData[, count_less_0:=tmpData]
tmpData <- apply(inputData[, -1, with=FALSE], 1, function(x) sum(x == 0))
inputData[, count_0:=tmpData]

#-------------------------------------------------------------------------------
# Split data into train and test
#-------------------------------------------------------------------------------
set.seed(1234)
random_splits <- runif(nrow(inputData))
train_df <- data.frame(inputData[random_splits < .5, ])
validate_df <- data.frame(inputData[random_splits >=.5, ])

outcome_name <- "QuoteConversion_Flag"
feature_names <- setdiff(names(inputData), outcome_name)

#-------------------------------------------------------------------------------
# Data quality check
#-------------------------------------------------------------------------------
# Proportion of 0/1 in three datasets
prop.table(table(inputData[, outcome_name, with=FALSE]))
prop.table(table(train_df[, outcome_name]))
prop.table(table(validate_df[, outcome_name]))

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
Version <- 1

for(Version in 1:35) {
  procStartTime <- Sys.time()
  cat("=========================================================\n")

  # List of remaining variables
  new_features <- setdiff(feature_names, selVars)

  # Generate different seeds for reducing the variance in the prediction
  seeds <- sample.int(4096, size=4) - 1

  # Initiate clusters
  cl <- makeCluster(12, type="SOCK")
  registerDoSNOW(cl)

  #-------------------------------------------------------------------------------
  # Forward selection
  #-------------------------------------------------------------------------------
  cat("Version: ", Version, " Features: ", length(new_features), "\n")

  # Initiate outputs
  forward_out <- NULL
  for(seed in seeds) {
    fs_start_time <- Sys.time()
    seedOut <- foreach(i=1:length(new_features), .inorder=FALSE, .packages=c("xgboost", "data.table", "caTools", "tcltk", "foreach")) %dopar% {
      if(!exists("pb")) pb <- tkProgressBar("Variables Completed", min=1, max=length(new_features))
      setTkProgressBar(pb, i)

      startTime <- Sys.time()
      tmp_features <- c(selVars, new_features[i])

      # Data for xgboost
      dtrain <- xgb.DMatrix(data=data.matrix(train_df[, tmp_features]), label=train_df[, outcome_name])
      dval <- xgb.DMatrix(data=data.matrix(validate_df[, tmp_features]))

      # Fit cross validation model using modified xgb.cv
      set.seed(seed)
      system.time(xgb_cv_new <- xgb.cv1(params=param0, data=dtrain, nrounds=nrounds,
        metrics=list("auc"), nfold=3, verbose=FALSE, prediction=TRUE, nthread=8))
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

      # Fit final model
      set.seed(seed)
      system.time(xgb_fit0 <- xgb.train(params=param0, data=dtrain, nrounds=new_nrounds, nthread=8)) # 14 Seconds
      tmpPred <- predict(object=xgb_fit0, newdata=dval, ntreelimit=new_nrounds)
      auc <- colAUC(X=tmpPred, y=validate_df[, outcome_name])

      # Return from the loop
      endTime <- Sys.time()
      timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
      retVal <- data.table(SelectedVars=new_features[i])
      retVal[, N:=i]
      retVal[, test_auc_mean:=xgb_cv_perf[tree==new_nrounds, test_auc_mean]]
      retVal[, test_auc_std:=xgb_cv_perf[tree==new_nrounds, test_auc_std]]
      retVal[, val_auc:=as.numeric(auc)]
      retVal[, time_secs:=timeTaken]
      retVal
    }
    fs_end_time <- Sys.time()
    fs_time <- round(as.numeric(difftime(fs_end_time, fs_start_time, units="secs")), 0)
    seedOut <- rbindlist(seedOut)
    seedOut[, TimeTaken:=fs_time]
    seedOut[, Seed:=seed]
    forward_out <- rbind(forward_out, seedOut)
    close(pb)
    rm(pb)
  }

  forward_out[, Max_Diff:=max(val_auc)-val_auc]
  setorder(forward_out, -test_auc_mean)
  write.csv(forward_out, paste0("C:/Users/Administrator/Dropbox/Public/Homesite/model/xgboost_cv/fs_both_direction_2/xgboost_cv_fs_2_", Version, ".csv"), row.names=FALSE)
  print(head(data.frame(forward_out)))

  #-------------------------------------------------------------------------------
  # Backward elimination
  #-------------------------------------------------------------------------------
  selVars <- c(selVars, forward_out[1, SelectedVars])
  new_features <- c(selVars, "")

  # Initiate outputs
  bs_start_time <- Sys.time()

  backward_out <- foreach(i=1:length(new_features), .inorder=FALSE, .packages=c("xgboost", "data.table", "caTools", "foreach")) %dopar% {
    startTime <- Sys.time()
    tmp_features <- setdiff(selVars, new_features[i])
    retVal <- data.table(DroppedVar=new_features[i])
    retVal[, N:=i]

    # Data for xgboost
    dtrain <- xgb.DMatrix(data=data.matrix(train_df[, tmp_features]), label=train_df[, outcome_name])
    dval <- xgb.DMatrix(data=data.matrix(validate_df[, tmp_features]))

    # Performance for different seeds
    out <- foreach(seed=seeds, .inorder=FALSE, .packages=c("xgboost", "data.table", "caTools")) %do% {
      # Fit cross validation model using modified xgb.cv
      set.seed(seed)
      system.time(xgb_cv_new <- xgb.cv1(params=param0, data=dtrain, nrounds=nrounds,
        metrics=list("auc"), nfold=3, verbose=FALSE, prediction=TRUE, nthread=8))
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

      # Fit final model
      set.seed(seed)
      system.time(xgb_fit0 <- xgb.train(params=param0, data=dtrain, nrounds=new_nrounds, nthread=8)) # 14 Seconds
      tmpPred <- predict(object=xgb_fit0, newdata=dval, ntreelimit=new_nrounds)
      auc <- colAUC(X=tmpPred, y=validate_df[, outcome_name])

      # Return from the loop
      tmpOut <- data.table(seed=seed)
      tmpOut[, test_auc_mean:=xgb_cv_perf[tree==new_nrounds, test_auc_mean]]
      tmpOut[, test_auc_std:=xgb_cv_perf[tree==new_nrounds, test_auc_std]]
      tmpOut[, val_auc:=as.numeric(auc)]
      tmpOut
    }
    endTime <- Sys.time()
    timeTaken <- round(as.numeric(difftime(endTime, startTime, units="secs")), 0)
    out <- rbindlist(out)
    out[, seed:=NULL]
    out <- out[, lapply(.SD, mean)]
    out[, time_secs:=timeTaken]

    retVal <- cbind(retVal, out)
    retVal
  }
  bs_end_time <- Sys.time()
  bs_time <- round(as.numeric(difftime(bs_end_time, bs_start_time, units="secs")), 0)
  backward_out <- rbindlist(backward_out)
  backward_out[, TimeTaken:=bs_time]
  backward_out[, Max_Diff:=max(val_auc)-val_auc]
  setorder(backward_out, -test_auc_mean)
  write.csv(backward_out, paste0("C:/Users/Administrator/Dropbox/Public/Homesite/model/xgboost_cv/fs_both_direction_2/xgboost_cv_bs_2_", Version, ".csv"), row.names=FALSE)
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
