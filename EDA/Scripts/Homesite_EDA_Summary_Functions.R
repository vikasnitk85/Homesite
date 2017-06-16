#-------------------------------------------------------------------------------
# Functions in this code are used to prepare the EDA report in the form of HTML
# Author - Vikas Agrawal
# Date Created - 05-August-2015
# Last Modified - 09-December-2015
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# sameLengthString()
# The function converts the string vector with different length to same length
# Author - Vikas Agrawal
#-------------------------------------------------------------------------------
sameLengthString <- function(x) {
  nSpace <- max(nchar(x)) - nchar(x)
  tempSpace <- NULL
  for(i in 1:length(x)) {
    tempSpace[i] <- paste(rep(" ", nSpace[i]), collapse="")
  }
  retVal <- paste0(tempSpace, x)
  retVal
}

#-------------------------------------------------------------------------------
# desStats()
# The function returns basic descriptive statistics for both numeric as well as
# categorical variable.
# Author - Vikas Agrawal
# Examples
#   desStats("Field6", train, nCats=20, freqCut=95/5, uniqueCut=10)
#   desStats("Field9", train, nCats=20, freqCut=95/5, uniqueCut=10)
#-------------------------------------------------------------------------------
desStats <- function(x, Data, nCats=20, freqCut=95/5, uniqueCut=10) {
  tempData <- Data[[x]]
  varType <- class(tempData)

  n <- length(tempData)                                                        # Number of observations
  nMiss <- sum(is.na(tempData))                                                # Number of missing observations
  nMissPer <- paste0(format(round(nMiss * 100/n, 2), 2), "%")                  # Missing percentage

  if(varType=="factor" | varType=="character") {
    tempData <- factor(tempData)                                               # Convert character variables to factor
    nFacs <- nlevels(tempData)                                                 # Number of factors
    tab <- sort(table(tempData), decreasing=TRUE)
    u2 <- round(1 - sum((tab/n)^2), 4)                                         # Variability factor

    if(length(tab) > nCats)
      tab <- tab[1:nCats]
    tabNames <- sameLengthString(names(tab))
    tabValues <- sameLengthString(format(tab, big.mark=",", scientific=FALSE))
    tabPct <- format(round(tab/n * 100, 2), nsmall=2)
    uniqueValues <- paste0(paste(tabNames, tabValues, tabPct, sep=" ## "), "%")

    n <- format(n, big.mark=",", scientific=FALSE)
    nMiss <- format(nMiss, big.mark=",", scientific=FALSE)
    nFacs <- format(nFacs, big.mark=",", scientific=FALSE)

    tempOutput <- sameLengthString(c(n, nMiss, nMissPer, nFacs, u2, uniqueValues))
    statNames <- c("Number of Observations", "Number of Missing",
      "Percent Missing", "Number of Factors", "Variability Factor",
      paste0("Unique Value ", 1:length(uniqueValues), " (class ## count ## pct_count)"))
    retVal <- data.frame(cbind(Statistics=statNames, Value=tempOutput),
      stringsAsFactors=FALSE)
  } else {
    nUnique <- length(unique(tempData[complete.cases(tempData)]))
    variance <- var(tempData, na.rm=TRUE)
    stdev <- sd(tempData, na.rm=TRUE)
    stats <- c(variance, stdev, summary(tempData)[1:6])
    # stats <- round(stats, 2)
    t <- table(tempData[!is.na(tempData)])
    if(length(t) <= 1) {
      freqRatio <- 0
    } else {
      w <- which.max(t)
      freqRatio <- max(t, na.rm=TRUE)/max(t[-w], na.rm=TRUE)
    }
    percentUnique <- 100 * nUnique/n
    zeroVar <- (nUnique==1) | all(is.na(tempData))
    u2 <- as.character((freqRatio > freqCut & percentUnique <=
      uniqueCut) | zeroVar)

    n <- format(n, big.mark=",", scientific=FALSE)
    nMiss <- format(nMiss, big.mark=",", scientific=FALSE)
    nUnique <- format(nUnique, big.mark=",", scientific=FALSE)
    stats <- format(stats, big.mark=",", scientific=FALSE)

    tempOutput <- c(n, nMiss, NA, nUnique, NA, stats)
    # tempOutput <- format(tempOutput, big.mark=",", scientific=FALSE)
    tempOutput[3] <- format(nMissPer, nsmall=2)
    tempOutput[5] <- u2
    tempOutput <- sameLengthString(tempOutput)
    names(tempOutput) <- NULL
    statNames <- c("Number of Observations", "Number of Missing",
      "Percentage Missing", "Number of Unique Values", "Zero or Near-Zero Variance",
      "Variance", "Standard Deviation", "Minimum", "1st Qu.", "Median",
      "Mean", "3rd Qu.", "Maximum")
    retVal <- data.frame(cbind(Statistics=statNames, Value=tempOutput),
      stringsAsFactors=FALSE)
  }
  retVal
}

#-------------------------------------------------------------------------------
# corAnalysis()
# The function perform correlation analysis for numeric variable with other
# numeric variables in the data and returns top correlated variables with
# correlation.
# Author - Vikas Agrawal
# Examples
#   corAnalysis("Field9", train, cutOff=0.7)
#-------------------------------------------------------------------------------
corAnalysis <- function(x, Data, cutOff=0.8) {
  numVars <- setdiff(names(Data)[which(sapply(Data, class) == "numeric" | sapply(Data, class) == "integer")], x)
  tempCorr <- cor(Data[[x]], Data[, numVars, with=FALSE], use="pairwise.complete.obs")[1,]
  tempCorr <- tempCorr[order(abs(tempCorr), decreasing=TRUE)]
  retVal <- tempCorr[which(abs(tempCorr) >= cutOff)]
  retVal <- format(round(retVal, 4), nsmall=4)
  retVal <- data.frame(cbind(Variable=names(retVal), Correlation=retVal),
    stringsAsFactors=FALSE)
  row.names(retVal) <- NULL
  retVal
}

#-------------------------------------------------------------------------------
# statsTest()
# The function gives statistical test based on the type of independent variable.
# Author - Vikas Agrawal
# Examples
#   statsTest("Field9", "QuoteConversion_Flag", train)
#   statsTest("Field6", "QuoteConversion_Flag", train)
#-------------------------------------------------------------------------------
statsTest <- function(x, y, Data) {
  dataX <- Data[[x]]
  classX <- class(dataX)

  dataY <- Data[[y]]
  dataY <- as.factor(dataY)

  if(classX=="factor" | classX=="character") {
    dataX <- as.factor(dataX)
    tempOut <- try(chisq.test(x=dataX, y=dataY))
    if(class(tempOut)!="try-error") {
      retVal <- matrix(NA, nrow=3, ncol=2)
      retVal[, 1] <- c("X-squared", "df", "p-value")
      retVal[, 2] <- format(round(c(tempOut$statistic, tempOut$parameter, tempOut$p.value), 4), big.mark=",", scientific=FALSE, nsmall=4)
      retVal <- data.frame(retVal, stringsAsFactors=FALSE)
      colnames(retVal) <- c("Statistics", "Value")
      rownames(retVal) <- NULL
    } else {
      retVal <- as.matrix("not enough finite observations!")
    }
  } else {
    tempOut <- try(t.test(as.formula(paste0(x, " ~ ", y)), data=Data))
    if(class(tempOut)!="try-error") {
      retVal <- matrix(NA, nrow=7, ncol=2)
      retVal[, 1] <- c("t", "df", "p-value", "95 percent lower confidence",
        "95 percent upper confidence", paste0("mean in group ", levels(dataY)))
      retVal[, 2] <- format(round(c(tempOut$statistic,
      tempOut$parameter, tempOut$p.value, as.numeric(tempOut$conf.int),
        as.numeric(tempOut$estimate)), 4), big.mark=",", scientific=FALSE, nsmall=4)
      retVal <- data.frame(retVal, stringsAsFactors=FALSE)
      colnames(retVal) <- c("Statistics", "Value")
      rownames(retVal) <- NULL
    } else {
      retVal <- as.matrix("not enough finite observations!")
    }
  }
  return(retVal)
}

#-------------------------------------------------------------------------------
# parseRpart()
# The function takes decision rules and return the rules in a data frame
# Author - Vikas Agrawal
#-------------------------------------------------------------------------------
parseRpart <- function(x, rpart.rules) {
  out_rule <- data.frame(min=numeric(0), max=numeric(0),
                min_comp=character(0), max_comp=character(0),
                tree_node = character(0), stringsAsFactors=F)

  for(i in seq_along(rpart.rules)) {
    t1 <- gsub("Data[[x]]", fixed=TRUE, replacement="", x=rpart.rules[[i]])
    ge <- gsub(pattern=">=", x=(t1[which(grepl("*>=[:digit:]*",x=t1,))]), fixed=T, replacement="")
    g  <- gsub(pattern=">" , x=(t1[which(grepl("*> [:digit:]*",x=t1,))]), fixed=T, replacement="")
    l  <- gsub(pattern="<" , x=(t1[which(grepl("*< [:digit:]*",x=t1,))]), fixed=T, replacement="")
    le <- gsub(pattern="<=", x=(t1[which(grepl("*<=[:digit:]*",x=t1,))]), fixed=T, replacement="")

    ge <- ifelse(length(ge)==0, NA, max(as.numeric(ge)))
    g  <- ifelse(length(g) ==0, NA, max(as.numeric(g)))
    le <- ifelse(length(le)==0, NA, min(as.numeric(le)))
    l  <- ifelse(length(l) ==0, NA, min(as.numeric(l)))

    out_rule[i, "min"] <- if (!(is.na(ge) && is.na(g))) max(ge, g, na.rm=T) else NA
    out_rule[i, "max"] <- if (!(is.na(le) && is.na(l))) max(le, l, na.rm=T) else NA
    out_rule[i, "min_comp"] <- ifelse(!is.na(ge), ">=", ifelse(!is.na(g), ">", ""))
    out_rule[i, "max_comp"] <- ifelse(!is.na(le), "<=", ifelse(!is.na(l), "<", ""))
    out_rule[i, "tree_node"] <- names(rpart.rules)[i]
  }
  out_rule
}

#-------------------------------------------------------------------------------
# ivNumDT()
# The function bins the numeric variable by decision tree rules
# Author - Vikas Agrawal
# Examples
#   Data <- ivNumDT("Field9", "QuoteConversion_Flag", train)
#-------------------------------------------------------------------------------
ivNumDT <- function(x, y, Data, rcontrol=NULL, x_new=paste0(x, "_DT")) {
  # Define rpart control parameters to run decision tree model
  if(is.null(rcontrol)) {
    rcontrol <- rpart.control(cp=10^-8, maxdepth=4)
  }

  # Build the decision tree model
  dtModel <- rpart(formula=as.numeric(Data[[y]]) ~ Data[[x]], data=Data, control=rcontrol)

  # Capture the components of decision tree
  rpart.rules <- path.rpart(dtModel, rownames(dtModel$frame)[dtModel$frame$var=="<leaf>"], print.it = FALSE)

  # Convert decision tree rules to data frame
  tree_rules <- parseRpart(x, rpart.rules)
  tree_rules <- tree_rules[order(tree_rules$max), ]

  if(nrow(tree_rules) > 1) {
    for(i in 1:nrow(tree_rules)) {
      tree_rules[i, "class_label"] <- paste0(
                  ifelse(tree_rules$min_comp[i] == ">=", "[", "("),
                  ifelse(is.na(tree_rules$min[i]), "", tree_rules$min[i]),
                  ", ",
                  ifelse(is.na(tree_rules$max[i]), "", tree_rules$max[i]),
                  ifelse(tree_rules$max_comp[i] == "<=", "]", ")")
                  )
    }

    # Add binned variable in the data
    for(rules in 1:nrow(tree_rules)) {
      if(tree_rules[rules, "min_comp"] == ">=" && tree_rules[rules, "max_comp"] == "<")
        Data[which(Data[[x]] >= tree_rules[rules, "min"] & Data[[x]] < tree_rules[rules, "max"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == ">" && tree_rules[rules, "max_comp"] == "<")
        Data[which(Data[[x]] > tree_rules[rules, "min"] & Data[[x]] < tree_rules[rules, "max"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == "" && tree_rules[rules, "max_comp"] == "<")
        Data[which(Data[[x]] < tree_rules[rules, "max"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == ">=" && tree_rules[rules, "max_comp"] == "<=")
        Data[which(Data[[x]] >= tree_rules[rules, "min"] & Data[[x]] <= tree_rules[rules, "max"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == ">" && tree_rules[rules, "max_comp"] == "<=")
        Data[which(Data[[x]] > tree_rules[rules, "min"] & Data[[x]] <= tree_rules[rules, "max"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == "" && tree_rules[rules, "max_comp"] == "<=")
        Data[which(Data[[x]] <= tree_rules[rules, "max"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == ">=" && tree_rules[rules, "max_comp"] == "")
        Data[which(Data[[x]] >= tree_rules[rules, "min"]), x_new:=tree_rules[rules, "class_label"]]
      if(tree_rules[rules, "min_comp"] == ">" && tree_rules[rules, "max_comp"] == "")
        Data[which(Data[[x]] > tree_rules[rules, "min"]), x_new:=tree_rules[rules, "class_label"]]
    }

    Data[[x_new]] <- factor(Data[, x_new], levels=tree_rules$class_label, ordered=TRUE)
    Data[, x_new:=NULL]
    Data[[x_new]] <- droplevels(Data[[x_new]])
  }
  return(Data)
}

#-------------------------------------------------------------------------------
# getIVtable()
# The function generate information value table.
# Author - Vikas Agrawal
# Examples
#   ivTable <- getIVtable("Field6", "QuoteConversion_Flag", train)
#-------------------------------------------------------------------------------
getIVtable <- function(x, y, Data) {
  # Count of good and bad in dependent variables
  tempTab <- table(Data[[y]])
  total_1 <- as.numeric(tempTab["1"])
  total_0 <- as.numeric(tempTab["0"])

  iv_data <- as.data.frame.matrix(table(Data[[x]], Data[[y]]))
  names(iv_data) <- c("count_0", "count_1")

  # Compute WOE and IV for each category of the variable
  iv_data <- within(iv_data, {
               class <- row.names(iv_data)
               variable <- x
               pct_0 <- count_0 / total_0
               pct_1 <- count_1 / total_1
               odds <-  pct_0 / pct_1
               woe <- log(odds)
               miv <- (pct_0 - pct_1) * woe
  })

  # Some group for outcome 0 might have zero counts. This will result in -Inf or Inf WOE. Replacing - ODDS=1, WoE=0, MIV=0.
  if(any(iv_data$count_0 == 0) | any(iv_data$count_1 == 0)) {
    iv_data$woe <- ifelse(is.infinite(iv_data$woe), 0, iv_data$woe)
    iv_data$miv <- ifelse(is.infinite(iv_data$miv), 0, iv_data$miv)
    iv_data$odds <-ifelse(is.infinite(iv_data$odds), 1, iv_data$odds)
  }

  # Add other statistics
  iv_data[, "conv_rate"] <- iv_data[, "count_1"] / rowSums(iv_data[, c("count_0", "count_1")])
  iv_data[, "pct_miv"] <- iv_data[, "miv"] / sum(iv_data[, "miv"])
  nObs <- sum(iv_data[, c("count_0", "count_1")])
  GP <- sum(iv_data[, "count_1"]) / nObs
  GB <- sum(iv_data[, "count_0"]) / nObs
  iv_data[, "pct_class"] <- rowSums(iv_data[, c("count_0", "count_1")]) / nObs
  iv_data[, "exp_good"] <- rowSums(iv_data[, c("count_0", "count_1")]) * GP
  iv_data[, "exp_bad"] <- rowSums(iv_data[, c("count_0", "count_1")]) * GB
  iv_data[, "chi_sq"] <- ((iv_data[, "count_1"] - iv_data[, "exp_good"]) ^ 2) / iv_data[, "exp_good"] +
    ((iv_data[, "count_0"] - iv_data[, "exp_bad"]) ^ 2) / iv_data[, "exp_bad"]

  # Reorder variables
  iv_data <- iv_data[c("variable", "class", "pct_class", "conv_rate", "woe", "miv", "pct_miv", "count_0", "count_1",
    "pct_0", "pct_1", "odds", "exp_good", "exp_bad", "chi_sq")]
  rownames(iv_data) <- NULL

  # Formatting Numbers
  iv_data[, "pct_class"] <- paste0(round(iv_data[, "pct_class"]*100, 2), "%")
  iv_data[, "conv_rate"] <- paste0(round(iv_data[, "conv_rate"]*100, 2), "%")
  iv_data[, "woe"] <- round(iv_data[, "woe"], 4)
  iv_data[, "miv"] <- round(iv_data[, "miv"], 4)
  iv_data[, "pct_miv"] <- paste0(round(iv_data[, "pct_miv"]*100, 2), "%")
  iv_data[, "count_0"] <- format(iv_data[, "count_0"], big.mark=",", scientific=FALSE)
  iv_data[, "count_1"] <- format(iv_data[, "count_1"], big.mark=",", scientific=FALSE)
  iv_data[, "pct_0"] <- paste0(round(iv_data[, "pct_0"]*100, 2), "%")
  iv_data[, "pct_1"] <- paste0(round(iv_data[, "pct_1"]*100, 2), "%")
  iv_data[, "odds"] <- round(iv_data[, "odds"], 2)
  iv_data[, "exp_good"] <- format(round(iv_data[, "exp_good"], 0), big.mark=",", scientific=FALSE)
  iv_data[, "exp_bad"] <- format(round(iv_data[, "exp_bad"], 0), big.mark=",", scientific=FALSE)
  iv_data[, "chi_sq"] <- format(round(iv_data[, "chi_sq"], 0), big.mark=",", scientific=FALSE)

  # Output
  iv_data
}

#-------------------------------------------------------------------------------
# woePlot()
# Author - Vikas Agrawal
# Examples
#   ivTable <- getIVtable("Field6", "QuoteConversion_Flag", train)
#   woePlot(ivTable)
#-------------------------------------------------------------------------------
woePlot <- function(iv) {
  woe <- NULL
  ggplot(data=iv) + geom_bar(aes(y=woe, x=class), stat="identity", position="identity") +
  theme(
    panel.grid.major.x=element_blank(),
    panel.grid.major.y=element_line(linetype="dashed", colour="grey"),
    panel.grid.minor=element_blank(),
    panel.background=element_blank(),
    axis.ticks.x=element_blank(),
    axis.ticks.y=element_blank(),
    axis.text.x=element_text(angle=45, hjust=1)
  ) +
  xlab("") +
  ylab("WoE") +
  ggtitle(paste0("WoE Pattern - ", iv$variable[1], "\nInformation Value: ", round(sum(iv[, "miv"]), 4)))
}

#-------------------------------------------------------------------------------
# uniDist()
# The function generate the R object for plotting univariate distribution (Bar Chart)
# Author - Vikas Agrawal
# Examples
#   uniDist("Field6", train, nCat=10, sorted=FALSE)
#-------------------------------------------------------------------------------
uniDist <- function(x, Data, nCat=10, sorted=FALSE) {
  tempData <- Data[[x]]
  tab <- table(tempData)
  tabPer <- paste0(round(tab * 100/length(tempData), 2), "%")
  Output <- data.frame(tab, tabPer)
  names(Output) <- c("Class", "Count", "Percentage")
  if(sorted) Output <- Output[order(Output$Count, decreasing=TRUE), ]
  if(nlevels(Output$Class) > nCat) Output <- Output[1:nCat, ]
  row.names(Output) <- NULL

  plotObj <- ggplot(data=Output, aes(x=Class, y=Count)) +
    geom_bar(stat="identity", fill="dodgerblue3", colour="dodgerblue3") +
    theme_bw() + guides(fill=FALSE) + ggtitle(paste0("Bar Chart - ",
    toupper(x))) + theme(plot.title=element_text(size=14,
    face="bold", color="black")) + theme(axis.title=element_text(size=12,
    face="bold", color="black")) + theme(axis.text=element_text(size=10,
    face="bold", color="black")) + geom_text(size=4,
    data=Output, aes(x=Class, y=Count, label=Percentage),
    vjust=-0.2, fontface="bold") + xlab("") + ylab("Frequency") +
    theme(axis.text.x=element_text(angle=22))
  return(plotObj)
}

#-------------------------------------------------------------------------------
# missDist()
# The function generate distribution of dependent variable for missing and non-missing population
# Author - Vikas Agrawal
#-------------------------------------------------------------------------------
missDist <- function(x, y, Data) {
  tempData <- Data[, c(x, y), with=FALSE]
  if(sum(is.na(tempData[[x]])) > 0) {
    tempData[[x]] <- ifelse(is.na(tempData[[x]]), 1, 0)
    tab <- round(prop.table(table(tempData), margin=1)*100, 2)
    retVal <- as.matrix.data.frame(tab)
    retVal <- data.frame(cbind(c("Missing", "Non-Missing"), retVal))
    colnames(retVal) <- c("Population", "Used", "New")
    retVal[, "Used"] <- paste0(retVal[, "Used"], "%")
    retVal[, "New"] <- paste0(retVal[, "New"], "%")
  } else {
    retVal <- NULL
  }
  retVal
}
