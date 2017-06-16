rm(list=ls(all=TRUE))
setwd("/home/agrav00/new_used")
options(scipen=999)

load("sample_data/TGT_TSPH_V1A_RD201408.RData")
names(rawData) <- toupper(names(rawData))

varList <- read.csv("eda/20150825/Selected_Variables_20150825.csv", stringsAsFactors=FALSE)
y <- "NVEH_NEW"
rawData <- rawData[, c(y, varList$Variable)]
save(rawData, file="sample_data/TGT_TSPH_V1A_RD201408_2.RData")
