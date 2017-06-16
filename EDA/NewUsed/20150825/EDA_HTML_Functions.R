#-------------------------------------------------------------------------------
# Function for generating HTML reports
# Author - Vikas Agrawal
# Date Created - 08-August-2015
# Last Modified - 25-August-2015
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# htmlReport()
# The function is required to create layout of HTML report
# Author - Vikas Agrawal
#-------------------------------------------------------------------------------
htmlReport <- function(x, outDesc, outUni, outMissDist, outCor, outStat, outIv, outWoe, outMissIv, outCramer) {
  # HTML Report Heading
  writeLines("\n")
  cat("# **", x, "** #", sep = "")
  writeLines("\n")
  cat("-----------------------------------\n")
  writeLines("\n")

  # Descriptive Statistics
  writeLines("\n")
  cat("## **Descriptive Statistics** ##", sep = "")
  writeLines("\n")
  print(kable(outDesc, format = "html", padding = 0, align='r'))
  writeLines("\n")

  # Univariate Distribution
  writeLines("\n")
  cat("## **Univariate Distribution** ##", sep = "")
  writeLines("\n")
  print(outUni)
  writeLines("\n")

  # Missing Distribution
  if(!is.null(outMissDist)) {
    writeLines("\n")
    cat("## **Distribution Missing/Non-Missing Population** ##", sep = "")
    writeLines("\n")
    print(kable(outMissDist, format = "html", padding = 0, align='r'))
    writeLines("\n")
  }

  # Correlation Analysis
  if(!is.null(outCor)) {
    writeLines("\n")
    cat("## **Correlation Analysis** ##", sep = "")
    writeLines("\n")
    if(nrow(outCor) > 0) {
      print(kable(outCor, format = "html", padding = 0, align='r'))
    } else {
      print(kable("No variables are correlated more than the specified cut-off", format = "pandoc", padding = 0))
    }
    writeLines("\n")
  }

  # Statistical Tests
  if(nrow(outStat) == 7) {
    writeLines("\n")
    cat("## **2 Independent Sample t-Test** ##", sep = "")
    writeLines("\n")
    print(kable(outStat, format = "html", padding = 0, align='r'))
    writeLines("\n")
  } else {
    writeLines("\n")
    cat("## **Chi-Squared Test** ##", sep = "")
    writeLines("\n")
    print(kable(outStat, format = "html", padding = 0, align='r'))
    writeLines("\n")
  }

  # Information Value Table & WOE
  if(length(outIv) > 1) {
    writeLines("\n")
    cat("## **Information Value Table by Decision Tree** ##", sep = "")
    writeLines("\n")
    cat("Information value = ", sum(outIv[[paste0(x, "_DT")]][, "miv"]))
    writeLines("\n")
    if(length(outMissIv) > 0) {
      cat("Information value by replacing missing observations with category MISSING = ", outMissIv[[paste0(x, "_DT")]])
    }
    if(nrow(outIv[[paste0(x, "_DT")]]) > 30) outIv[[paste0(x, "_DT")]] <- outIv[[paste0(x, "_DT")]][1:30, ]
    writeLines("\n")
    print(kable(outIv[[paste0(x, "_DT")]][-1], format = "html", padding = 0, align='r'))
    writeLines("\n")

    writeLines("\n")
    cat("## **Weight-of-Evidence Plot by Decision Tree** ##", sep = "")
    writeLines("\n")
    print(outWoe[[paste0(x, "_DT")]])
    writeLines("\n")

    writeLines("\n")
    cat("## **Information Value for Monotonic WOE** ##", sep = "")
    writeLines("\n")
    cat("Information value = ", sum(outIv[[paste0(x, "_MON")]][, "miv"]))
    writeLines("\n")
    if(length(outMissIv) > 0) {
      cat("Information value by replacing missing observations with category MISSING = ", outMissIv[[paste0(x, "_MON")]])
    }
    if(nrow(outIv[[paste0(x, "_MON")]]) > 30) outIv[[paste0(x, "_MON")]] <- outIv[[paste0(x, "_MON")]][1:30, ]
    writeLines("\n")
    print(kable(outIv[[paste0(x, "_MON")]][-1], format = "html", padding = 0, align='r'))
    writeLines("\n")

    writeLines("\n")
    cat("## **Monotonic WOE** ##", sep = "")
    writeLines("\n")
    print(outWoe[[paste0(x, "_MON")]])
    writeLines("\n")
  } else {
    writeLines("\n")
    cat("## **Information Value Table** ##", sep = "")
    writeLines("\n")
    cat("Information value = ", sum(outIv[[1]][, "miv"]))
    writeLines("\n")
    if(length(outMissIv) > 0) {
      cat("Information value by replacing missing observations with category MISSING = ", outMissIv[[1]])
    }
    writeLines("\n")
    if(nrow(outIv[[1]]) > 30) outIv[[1]] <- outIv[[1]][1:30, ]
    print(kable(outIv[[1]][-1], format = "html", padding = 0, align='r'))
    writeLines("\n")

    writeLines("\n")
    cat("## **Weight-of-Evidence Plot** ##", sep = "")
    writeLines("\n")
    print(outWoe[[1]])
    writeLines("\n")
  }

  # Association Analysis
  if(length(outCramer) > 1) {
    writeLines("\n")
    cat("## **Association Analysis for DT Variable** ##", sep = "")
    writeLines("\n")
    print(kable(outCramer[[paste0(x, "_DT")]][1:6, ], format = "html", padding = 0, align='r'))
    writeLines("\n")

    writeLines("\n")
    cat("## **Association Analysis for Monotonic Variable** ##", sep = "")
    writeLines("\n")
    print(kable(outCramer[[paste0(x, "_MON")]][1:6, ], format = "html", padding = 0, align='r'))
    writeLines("\n")
  } else {
    writeLines("\n")
    cat("## **Association Analysis** ##", sep = "")
    writeLines("\n")
    print(kable(outCramer[[1]][1:6, ], format = "html", padding = 0, align='r'))
    writeLines("\n")
  }
}

#-------------------------------------------------------------------------------
# genHTMLReport()
# The function generates HTML report
# Author - Vikas Agrawal
#-------------------------------------------------------------------------------
genHTMLReport <- function(x, outDesc, outUni, outMissDist, outCor, outStat, outIv, outWoe, outMissIv, outCramer, outFile) {
  knitr::opts_chunk$set(echo = FALSE, comment = NA, message = FALSE,
    strip.white = TRUE, split = TRUE, warning = FALSE, results = "asis",
    fig.align = "center", fig.height = 8, fig.width = 16)
  tmpWtite <- c("---", "output:", "  knitrBootstrap::bootstrap_document:",
    "    theme: Flatly", "    highlight: Xcode", "    theme.chooser: FALSE",
    "    highlight.chooser: FALSE", "    menu: FALSE", "---",
    "", "``` {r Plot_Charts_New, echo=FALSE, message=FALSE, warning=FALSE}",
    "htmlReport(x, outDesc, outUni, outMissDist, outCor, outStat, outIv, outWoe, outMissIv, outCramer)",
    "```", "", "***", "", "<img src='https://cdn.ihs.com/www2/a/p/media/images/icons/mstile-144x144.png' width='80px' height='80px' />",
    "***Author: Vikas Agrawal, Kumar Manglam Thakur***")
  writeLines(tmpWtite, "htmlReport.Rmd")
  render("htmlReport.Rmd", "knitrBootstrap::bootstrap_document",
    output_file = outFile)
  if(file.exists("htmlReport.Rmd"))
    file.remove("htmlReport.Rmd")
}
