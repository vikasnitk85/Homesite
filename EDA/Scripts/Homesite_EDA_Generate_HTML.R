#-------------------------------------------------------------------------------
# Generate HTML reports from EDA Summary
# Author - Vikas Agrawal
# Date Created - 08-August-2015
# Last Modified - 25-August-2015
#-------------------------------------------------------------------------------

rm(list=ls(all=TRUE))
setwd("D:/Projects/Autos/TrueSource/New_Used")
options(scipen=999)

library(knitrBootstrap)
library(knitr)
library(rmarkdown)

load("Development/EDA/2015-08-25/EDA_SUMMARY_V1A_RD201408_20150825.RData")
source("VersionControl/EDA/20150825/EDA_HTML_Functions.R")
varList <- names(retDesStat)

# Create Folders to Store Outputs
dir.create(file.path(getwd(), "Development/EDA/2015-08-25/Reports"), showWarnings = FALSE)

#-------------------------------------------------------------------------------
# Generate HTML Reports
#-------------------------------------------------------------------------------
startTime <- Sys.time()
for(x in varList) {
  source("VersionControl/EDA/20150825/EDA_HTML_Functions.R")
  outDesc <- retDesStat[[x]]
  outUni <- retUniDist[[x]]
  outMissDist <- retMissDist[[x]]
  outCor <- retCorAna[[x]]
  outStat <- retStatOut[[x]]
  outIv <- retIV[grep(x, names(retIV))]
  if(length(outIv) > 1 & length(grep("_DT", names(outIv))) == 0) outIv[[setdiff(names(outIv), x)]] <- NULL
  outWoe <- retWOE[grep(x, names(retWOE))]
  if(length(outWoe) > 1 & length(grep("_DT", names(outWoe))) == 0) outWoe[[setdiff(names(outWoe), x)]] <- NULL
  outMissIv <- retMissIV[grep(x, names(retMissIV))]
  if(length(outMissIv) > 1 & length(grep("_DT", names(outMissIv))) == 0) outMissIv[[setdiff(names(outMissIv), x)]] <- NULL
  outCramer <- retCramersV[grep(x, names(retCramersV))]
  if(length(outCramer) > 1 & length(grep("_DT", names(outCramer))) == 0) outCramer[[setdiff(names(outCramer), x)]] <- NULL

  genHTMLReport(x, outDesc, outUni, outMissDist, outCor, outStat, outIv, outWoe, outMissIv, outCramer,
    outFile=paste0("Development/EDA/2015-08-25/Reports/EDA_", x, ".html"))
}
endTime <- Sys.time()
difftime(endTime, startTime)
