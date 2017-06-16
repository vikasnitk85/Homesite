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

validate_predictions <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
library(pROC)
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# H2o - Find Anomalies
#-------------------------------------------------------------------------------
localH2O = h2o.init()
homesite.hex<-as.h2o(train_df, destination_frame="train.hex")

homesite.dl = h2o.deeplearning(x = feature_names, training_frame = homesite.hex,
                               autoencoder = TRUE,
                               reproducible = T,
                               seed = 1234,
                               hidden = c(6,5,6), epochs = 50)
							   
homesite.anon = h2o.anomaly(homesite.dl, homesite.hex, per_feature=FALSE)
head(homesite.anon)
err <- as.data.frame(homesite.anon)
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')

#-------------------------------------------------------------------------------
# Modeling Without Anomalies (RandomForest)
#-------------------------------------------------------------------------------
# rebuild train_df_auto with best observations
train_df_auto <- train_df[err$Reconstruction.MSE < 0.09, ]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_known <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_known[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# Modeling Without Anomalies (RandomForest)
#-------------------------------------------------------------------------------
# rebuild train_df_auto with best observations
train_df_auto <- train_df[err$Reconstruction.MSE >= 0.09, ]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_unknown <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_unknown[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# Ensemble Prediction
#-------------------------------------------------------------------------------
valid_all <- (validate_predictions_known[,2] + validate_predictions_unknown[,2]) / 2

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=valid_all)

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# Benchmark xgboost
#-------------------------------------------------------------------------------
set.seed(1234)
h <- sample(nrow(train_df), 2600)

dval <- xgb.DMatrix(data=data.matrix(train_df[h, feature_names]), label=train_df[h, outcome_name])
dtrain <- xgb.DMatrix(data=data.matrix(train_df[-h, feature_names]), label=data.matrix(train_df[-h, outcome_name]))

watchlist<-list(val=dval, train=dtrain)
param <- list(objective           = "binary:logistic", 
              booster = "gbtree",
              eval_metric = "auc",
              eta                 = 0.02, # 0.06, #0.01,
              max_depth           = 10, #changed from default of 8
              subsample           = 0.9, # 0.7
              colsample_bytree    = 0.9 # 0.7
              #num_parallel_tree   = 2
              # alpha = 0.0001, 
              # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 100, 
                    verbose             = 2,  #1
                    # early.stop.round    = 150,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

pred <- predict(clf, data.matrix(validate_df[,feature_names]))
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=pred)
plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3))) #0.9592
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# Modeling Without Anomalies (xgboost)
#-------------------------------------------------------------------------------
# rebuild train_df_auto with best observations
train_df_auto <- train_df[err$Reconstruction.MSE < 0.09, ]

set.seed(1234)
h <- sample(nrow(train_df_auto), round(0.1*nrow(train_df_auto), 0))

dval <- xgb.DMatrix(data=data.matrix(train_df_auto[h, feature_names]), label=train_df_auto[h, outcome_name])
dtrain <- xgb.DMatrix(data=data.matrix(train_df_auto[-h, feature_names]), label=data.matrix(train_df_auto[-h, outcome_name]))

watchlist<-list(val=dval, train=dtrain)
param <- list(objective           = "binary:logistic", 
              booster = "gbtree",
              eval_metric = "auc",
              eta                 = 0.02, # 0.06, #0.01,
              max_depth           = 10, #changed from default of 8
              subsample           = 0.9, # 0.7
              colsample_bytree    = 0.9 # 0.7
              #num_parallel_tree   = 2
              # alpha = 0.0001, 
              # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 100, 
                    verbose             = 2,  #1
                    # early.stop.round    = 150,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

pred1 <- predict(clf, data.matrix(validate_df[,feature_names]))
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=pred1)
plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3))) #0.959
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# Modeling Wit Anomalies (xgboost)
#-------------------------------------------------------------------------------
# rebuild train_df_auto with best observations
train_df_auto <- train_df[err$Reconstruction.MSE >= 0.09, ]

set.seed(1234)
h <- sample(nrow(train_df_auto), round(0.1*nrow(train_df_auto), 0))

dval <- xgb.DMatrix(data=data.matrix(train_df_auto[h, feature_names]), label=train_df_auto[h, outcome_name])
dtrain <- xgb.DMatrix(data=data.matrix(train_df_auto[-h, feature_names]), label=data.matrix(train_df_auto[-h, outcome_name]))

watchlist<-list(val=dval, train=dtrain)
param <- list(objective           = "binary:logistic", 
              booster = "gbtree",
              eval_metric = "auc",
              eta                 = 0.02, # 0.06, #0.01,
              max_depth           = 10, #changed from default of 8
              subsample           = 0.9, # 0.7
              colsample_bytree    = 0.9 # 0.7
              #num_parallel_tree   = 2
              # alpha = 0.0001, 
              # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 100, 
                    verbose             = 2,  #1
                    # early.stop.round    = 150,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

pred2 <- predict(clf, data.matrix(validate_df[,feature_names]))
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=pred2)
plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')

#-------------------------------------------------------------------------------
# Ensemble Prediction
#-------------------------------------------------------------------------------
valid_all <- (pred1 + pred2) / 2

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=valid_all)

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')
