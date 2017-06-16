#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

# Define library path and add it to default library path
libPath <- "E:/R-Packages"
.libPaths(libPath)

# Load required libraries
pkgs <- c("caret", "data.table", "R.utils", "doSNOW", "h2o", "randomForest", "xgboost", "pROC")
sapply(pkgs, require, character.only=TRUE)

# Define the path for base directory and set it as a working directory
basePath <- "D:/Vikas/Homesite/"
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

set.seed(1234)
random_splits <- runif(nrow(inputData))
train_df <- data.frame(inputData[random_splits < .5, ])
dim(train_df)
validate_df <- data.frame(inputData[random_splits >=.5, ])
dim(validate_df)

outcome_name <- "QuoteConversion_Flag"
feature_names <- setdiff(names(train_df), outcome_name)

prop.table(table(inputData[, outcome_name, with=FALSE]))
prop.table(table(train_df[, outcome_name]))
prop.table(table(validate_df[, outcome_name]))

#-------------------------------------------------------------------------------
# Benchmark Random Forest
#-------------------------------------------------------------------------------
set.seed(1234)
rf_model <- randomForest(x=train_df[,feature_names],
                         y=as.factor(train_df[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)


