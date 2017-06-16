#-------------------------------------------------------------------------------
# Generate summary of the independent variables required for HTML report creation
# Author - Vikas Agrawal
# Date Created - 05-August-2015
# Last Modified - 25-August-2015
#-------------------------------------------------------------------------------

# Environment set-up
rm(list=ls(all=TRUE))
setwd("/home/agrav00/new_used")
options(scipen=999)

library(rpart)
library(ggplot2)
library(lsr)
library(data.table)

load("sample_data/TGT_TSPH_V1A_RD201408_2.RData")
y <- "NVEH_NEW"
varList <- setdiff(names(rawData), y)
numVars <- names(rawData)[which(sapply(rawData, class)=="numeric")]

# Source R codes for generating summary
source("eda/20150825/EDA_Summary_Functions.R")

#-------------------------------------------------------------------------------
# Generate Descriptive Statistics
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retDesStat <- list()
for(x in varList) {
  print(paste0(x, " - ", which(varList %in% x)))
  retDesStat[[x]] <- desStats(x, rawData, nCats=20, freqCut=95/5, uniqueCut=10)
}
endTime <- Sys.time()
timeTaken1 <- difftime(endTime, startTime)
timeTaken1 #1.779259 mins

#-------------------------------------------------------------------------------
# Perform Correlation Analysis
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retCorAna <- list()
for(x in numVars) {
  print(paste0(x, " - ", which(numVars %in% x)))
  retCorAna[[x]] <- corAnalysis(x, rawData, cutOff=0.7)
}
endTime <- Sys.time()
timeTaken2 <- difftime(endTime, startTime)
timeTaken2 #2.576793 mins

#-------------------------------------------------------------------------------
# Perform Statistical Analysis
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retStatOut <- list()
for(x in varList) {
  print(paste0(x, " - ", which(varList %in% x)))
  retStatOut[[x]] <- statsTest(x, y, rawData)
}
endTime <- Sys.time()
timeTaken3 <- difftime(endTime, startTime)
timeTaken3 #2.894253 mins

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
timeTaken4 #12.14533 mins

#-------------------------------------------------------------------------------
# Generate Monotonic Bins using Binned Variables by Decision Tree
#-------------------------------------------------------------------------------
startTime <- Sys.time()
for(x in numVars) {
  print(paste0(x, " - ", which(numVars %in% x)))
  Data[, paste0(x, "_MON")] <- monotoneIV(paste0(x, "_DT"), y, Data, maxCat=10)
}
endTime <- Sys.time()
timeTaken5 <- difftime(endTime, startTime)
timeTaken5 #48.77389 mins

#-------------------------------------------------------------------------------
# Generate Information Value Table and WOE
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retIV <- list()
retWOE <- list()
catVars <- setdiff(names(Data)[which(sapply(Data, function(x) class(x)[[1]])=="factor" | sapply(Data, function(x) class(x)[[1]])=="ordered")], y)
for(x in catVars) {
  print(paste0(x, " - ", which(catVars %in% x)))
  tempIV <- getIVtable(x, y, Data)
  if(length(grep("_DT", x)) > 0 | length(grep("_MON", x)) > 0) {
     tempIV$class <- factor(tempIV$class, levels=tempIV$class, ordered=TRUE)
  }
  retIV[[x]] <- tempIV
  retWOE[[x]] <- woePlot(tempIV)
}
endTime <- Sys.time()
timeTaken6 <- difftime(endTime, startTime)
timeTaken6 #1.418765 mins

#-------------------------------------------------------------------------------
# IV if missing observations replaced by MISSING category
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retMissIV <- list()
for(x in catVars) {
  print(paste0(x, " - ", which(catVars %in% x)))
  tmpData <- Data[, c(x, y)]
  if(sum(is.na(tmpData[, x])) > 0) {
    levels(tmpData[, x]) <- c(levels(tmpData[, x]), "MISSING")
    tmpData[is.na(tmpData[, x]), x] <- "MISSING"
    tempIV <- getIVtable(x, y, tmpData)
    retMissIV[[x]] <- sum(tempIV[, "miv"])
  }
}
endTime <- Sys.time()
timeTaken7 <- difftime(endTime, startTime)
timeTaken7 #1.288139 mins

#-------------------------------------------------------------------------------
# Generate Univariate Distribution
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retUniDist <- list()
for(x in varList) {
  print(paste0(x, " - ", which(varList %in% x)))
  if(class(Data[, x]) == "numeric") {
    retUniDist[[x]] <- uniDist(paste0(x, "_DT"), Data, nCat=20, sorted=FALSE)
  } else {
    retUniDist[[x]] <- uniDist(x, Data, nCat=20, sorted=TRUE)
  }
}
endTime <- Sys.time()
timeTaken8 <- difftime(endTime, startTime)
timeTaken8 #24.09392 secs

#-------------------------------------------------------------------------------
# Generate distribution of dependent variable for missing and non-missing population
#-------------------------------------------------------------------------------
startTime <- Sys.time()
retMissDist <- list()
for(x in varList) {
  print(paste0(x, " - ", which(varList %in% x)))
  retMissDist[[x]] <- missDist(x, y, rawData)
}
endTime <- Sys.time()
timeTaken9 <- difftime(endTime, startTime)
timeTaken9 #1.404839 mins

#-------------------------------------------------------------------------------
# Association Analysis
#-------------------------------------------------------------------------------
# save(Data, file="sample_data/TGT_TSPH_V1A_RD201408_3.RData")
startTime <- Sys.time()
Data <- Data[, c(y, catVars)]
retCramersV <- list()
for(x in catVars) {
  print(paste0(x, " - ", which(catVars %in% x)))
  tempVy <- format(round(cramersV(Data[, x], Data[, y]), 4), nsmall=4)
  tmpVx <- try(sapply(Data[, setdiff(names(Data), c(x, y))],
                      function(vrb) {
                        tmp <- try(cramersV(vrb, y=Data[, x]))
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
timeTaken10 <- difftime(endTime, startTime)
timeTaken10 #3.638435 hours

save(retDesStat, retCorAna, retStatOut, retIV, retWOE, retMissIV, retMissDist,
  retUniDist, retCramersV, file="eda/20150825/EDA_SUMMARY_V1A_RD201408_20150825.RData")
