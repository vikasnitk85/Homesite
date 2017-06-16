#-------------------------------------------------------------------------------
# Generate summary of the independent variables required for HTML report creation
# Author - Vikas Agrawal
# Date Created - 05-August-2015
# Last Modified - 09-December-2015
#-------------------------------------------------------------------------------

# Environment set-up
rm(list=ls(all=TRUE))
setwd("F:/Education/Kaggle/Homesite Quote Conversion")
options(scipen=999)

library(rpart)
library(ggplot2)
library(lsr)
library(data.table)
library(doSNOW)

# Register cores for parallel computing
cl <- makeCluster(10, type="SOCK")
registerDoSNOW(cl)

rawData <- fread("input/train.csv")
y <- "QuoteConversion_Flag"
varList <- setdiff(names(rawData), c(y, "QuoteNumber", "Original_Quote_Date"))
numVars <- setdiff(names(rawData)[which(sapply(rawData, class)=="numeric" | sapply(rawData, class)=="integer")], c(y, "QuoteNumber"))

# Source R codes for generating summary
source("EDA/Scripts/Homesite_EDA_Summary_Functions.R")

#-------------------------------------------------------------------------------
# Generate Descriptive Statistics
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retDesStat <- foreach(x=varList, .inorder=TRUE, .packages=c("data.table")) %dopar% {
  desStats(x, rawData, nCats=20, freqCut=95/5, uniqueCut=10)
}
names(retDesStat) <- varList
endTime <- Sys.time()
timeTaken1 <- difftime(endTime, startTime)
timeTaken1 #14.24104 secs

#-------------------------------------------------------------------------------
# Perform Correlation Analysis
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retCorAna <- foreach(x=numVars, .inorder=TRUE, .packages=c("data.table")) %dopar% {
  corAnalysis(x, rawData, cutOff=0.7)
}
names(retCorAna) <- numVars
endTime <- Sys.time()
timeTaken2 <- difftime(endTime, startTime)
timeTaken2 #1.292298 mins

#-------------------------------------------------------------------------------
# Perform Statistical Analysis
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retStatOut <- foreach(x=varList, .inorder=TRUE, .packages=c("data.table")) %dopar% {
  statsTest(x, y, rawData)
}
names(retStatOut) <- varList
endTime <- Sys.time()
timeTaken3 <- difftime(endTime, startTime)
timeTaken3 #26.1705 secs

#-------------------------------------------------------------------------------
# Generate Binned Variables using Decision Tree
#-------------------------------------------------------------------------------
startTime <- Sys.time()
Data <- rawData
for(x in numVars) {
  print(paste0(x, " - ", which(numVars %in% x)))
  Data <- ivNumDT(x, y, Data)
}
endTime <- Sys.time()
timeTaken4 <- difftime(endTime, startTime)
timeTaken4 #3.388554 mins

#-------------------------------------------------------------------------------
# Generate Information Value Table and WOE
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retIV <- list()
retWOE <- list()
catVars <- setdiff(names(Data)[which(sapply(Data, function(x) class(x)[[1]])=="factor" |
								     sapply(Data, function(x) class(x)[[1]])=="ordered" |
									 sapply(Data, function(x) class(x)[[1]])=="character")], y)
catVars <- setdiff(catVars, "Original_Quote_Date")

for(x in catVars) {
  print(paste0(x, " - ", which(catVars %in% x)))
  tempIV <- getIVtable(x, y, Data)
  if(length(grep("_DT", x)) > 0) {
     tempIV$class <- factor(tempIV$class, levels=tempIV$class, ordered=TRUE)
  }
  retIV[[x]] <- tempIV
  retWOE[[x]] <- woePlot(tempIV)
}
endTime <- Sys.time()
timeTaken5 <- difftime(endTime, startTime)
timeTaken5 #1.754024 mins

#-------------------------------------------------------------------------------
# Generate Univariate Distribution
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retUniDist <- list()
for(x in varList) {
  print(paste0(x, " - ", which(varList %in% x)))
  if(x %in% numVars & !is.null(Data[[paste0(x, "_DT")]])) {
    retUniDist[[x]] <- uniDist(paste0(x, "_DT"), Data, nCat=20, sorted=FALSE)
  } else {
    retUniDist[[x]] <- uniDist(x, Data, nCat=20, sorted=TRUE)
  }
}
endTime <- Sys.time()
timeTaken6 <- difftime(endTime, startTime)
timeTaken6 #9.51657 secs

#-------------------------------------------------------------------------------
# Association Analysis
#-------------------------------------------------------------------------------
startTime <- Sys.time()
Data <- Data[, c(y, catVars), with=FALSE]
retCramersV <- list()
for(x in catVars) {
  print(paste0(x, " - ", which(catVars %in% x)))
  tempVy <- format(round(cramersV(Data[[x]], Data[[y]]), 4), nsmall=4)
  tmpVx <- try(sapply(Data[, setdiff(names(Data), c(x, y)), with=FALSE],
                      function(vrb) {
                        tmp <- try(cramersV(vrb, y=Data[[x]]))
                        ifelse(class(tmp) != "try-error", tmp, NA)
                      }
               ))
  if(class(tmpVx) != "try-error") {
    tmpVx <- format(round(tmpVx, 4), nsmall=4)
    tmpV <- as.matrix(c(NA, sort(tmpVx, decreasing=TRUE)), ncol=1)
    tmpV <- data.frame(rownames(tmpV), tmpV, stringsAsFactors=FALSE)
    tmpV[1, ] <- c(y, tempVy)
    rownames(tmpV) <- NULL
    names(tmpV) <- c("Variable", "Cramer's V")
    retCramersV[[x]] <- tmpV
  }
}
endTime <- Sys.time()
timeTaken7 <- difftime(endTime, startTime)
timeTaken7 #1.082871 hours

save(retDesStat, retCorAna, retStatOut, retIV, retWOE,
  retUniDist, retCramersV, file="EDA/Ouptuts/20151210/EDA_SUMMARY_20151210.RData")
