#-------------------------------------------------------------------------------
# Environment Set-up
#-------------------------------------------------------------------------------
rm(list=ls(all=TRUE))
gc()

# Define library path and add it to default library path
libPath <- "E:/R-Packages"
.libPaths(libPath)

# Load required libraries
pkgs <- c("caret", "data.table", "R.utils", "doSNOW", "h2o", "randomForest", "xgboost", "pROC", "caretEnsemble")
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

#-------------------------------------------------------------------------------
# Split data into train and test
#-------------------------------------------------------------------------------
outcome_name <- "QuoteConversion_Flag"
feature_names <- setdiff(names(inputData), outcome_name)

set.seed(1234)
random_splits <- runif(nrow(inputData))
train_df <- inputData[random_splits < .5, ]
train_df <- data.frame(train_df)
train_df[, outcome_name] <- ifelse(train_df[, outcome_name]==0, "N", "Y")
train_df[, outcome_name] <- as.factor(train_df[, outcome_name])
dim(train_df)
validate_df <- data.frame(inputData[random_splits >=.5, ])
dim(validate_df)


prop.table(table(inputData[, outcome_name, with=FALSE]))
prop.table(table(train_df[, outcome_name]))
prop.table(table(validate_df[, outcome_name]))

#-------------------------------------------------------------------------------
# Fit models
#-------------------------------------------------------------------------------
Methods <- c("avNNet", "bagEarth", "bagEarthGCV", "bayesglm", "bdk", "cforest",
  "ctree", "ctree2", "dnn", "dwdLinear", "dwdRadial", "earth", "evtree", "extraTrees",
  "gamLoess", "gamSpline", "glm", "glmboost", "glmnet", "glmStepAIC", "gpls", "hda", "hdda",
  "knn", "lda", "LMT", "LogitBoost", "mda", "mlpWeightDecay", "multinom", "nb", "nnet",
  "oblique.tree", "OneR", "pam", "parRF", "pcaNNet", "pda", "pda2", "plr", "pls", "plsRglm",
  "ranger", "rda", "rf", "rotationForest", "rotationForestCp", "rpart", "RRF", "RRFglobal",
  "rrlda", "sda", "spls", "stepLDA", "stepQDA", "svmLinear", "svmLinear2", "svmPoly", "svmRadial",
  "svmRadialCost", "svmRadialWeights", "xgbLinear", "xgbTree", "xyf")

# train_df <- train_df[1:100, 1:10]
# feature_names <- setdiff(names(train_df), outcome_name)

feature_names <- setdiff(feature_names, names(which(apply(train_df[, feature_names], 2, sd)==0)))
TimeSheet <- data.table(Methods=Methods, Time=0)

set.seed(102)
index <- createFolds(train_df[, outcome_name], k = 3)
ctrl <- trainControl(method="cv", number=3, repeats=1, savePredictions=TRUE, verboseIter=FALSE, index=index, 
  classProbs=TRUE, summaryFunction=twoClassSummary)

model_list_big <- list()
loadedObjects <- ls()
for(Method in Methods) {
  cl <- makeCluster(3, type="SOCK")
  registerDoSNOW(cl)

  print(Method)
  a <- Sys.time()
  set.seed(153)
  model_list_big[[Method]] <- try(caret:::train(x=train_df[, feature_names], y=train_df[, outcome_name],
      method=Method, trControl=ctrl, tuneLength=3, metric="ROC"))
  print(model_list_big[[Method]])
  b <- Sys.time()
  TimeSheet[Methods==Method, Time:=round(as.numeric(difftime(b, a, units="secs")), 0)]
  print(data.frame(TimeSheet[Methods==Method, ]))
  stopCluster(cl)
  Sys.sleep(30)
  rm(list=setdiff(ls(all=TRUE), c(loadedObjects, "Method")))
  gc()
}

# set.seed(153)
# model_list_big <- caretList(
  # QuoteConversion_Flag~., data=train_df,
  # trControl=ctrl,
  # methodList=Methods,
  # metric="ROC"
# )


table(unlist(lapply(model_list_big, class)))
modelPerf <- rbindlist(lapply(model_list_big, getTrainPerf))
setorder(modelPerf, TrainROC)

corMat <- modelCor(resamples(model_list_big))
findCorrelation(corMat, cutoff = .90, names = TRUE, verbose = FALSE)

