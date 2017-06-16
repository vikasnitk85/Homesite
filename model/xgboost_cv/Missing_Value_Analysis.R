rm(list=ls(all=TRUE))
gc()
library(data.table)
basePath <- "F:/Education/Kaggle/Homesite/"
setwd(basePath)
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

inputData[is.na(PropertyField29), PropertyField29:=-1]
inputData[is.na(PersonalField84), PersonalField84:=-1]

negativeVal <- NULL
uniqueValues <- NULL
for(f in names(inputData)) {
  # print(which(names(inputData) %in% f))
  tmpData <- inputData[[f]]
  negativeVal[f] <- sum(tmpData < 0)
  uniqueValues[f] <- ifelse(negativeVal[f] > 0, length(unique(tmpData)) - 1, length(unique(tmpData)))
}

# negativeVal <- negativeVal[negativeVal!=0]
# negativeVal <- sort(negativeVal, decreasing=TRUE)
# uniqueValues <- uniqueValues[names(negativeVal)]

# for(f in names(negativeVal)) {
  # cat("================================\n")
  # print(f)
  # print(table(inputData[[f]]))
# }

finalOut <- data.table(Var=names(negativeVal), NegativeCount=negativeVal, UniqueCount=uniqueValues)
write.csv(finalOut, "model/xgboost_cv/Missing_Value_Analysis.csv", row.names=FALSE)
