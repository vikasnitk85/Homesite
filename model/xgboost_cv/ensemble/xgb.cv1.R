xgb.cv1 <- function (params = list(), data, nrounds, nfold, label = NULL,
    missing = NULL, prediction = FALSE, showsd = TRUE, metrics = list(),
    obj = NULL, feval = NULL, stratified = TRUE, folds = NULL,
    verbose = T, print.every.n = 1L, early.stop.round = NULL,
    maximize = NULL, ...) {
    if (typeof(params) != "list") {
        stop("xgb.cv: first argument params must be list")
    }
    if (!is.null(folds)) {
        if (class(folds) != "list" | length(folds) < 2) {
            stop("folds must be a list with 2 or more elements that are vectors of indices for each CV-fold")
        }
        nfold <- length(folds)
    }
    if (nfold <= 1) {
        stop("nfold must be bigger than 1")
    }
    if (is.null(missing)) {
        dtrain <- xgb.get.DMatrix(data, label)
    }
    else {
        dtrain <- xgb.get.DMatrix(data, label, missing)
    }
    dot.params = list(...)
    nms.params = names(params)
    nms.dot.params = names(dot.params)
    if (length(intersect(nms.params, nms.dot.params)) > 0)
        stop("Duplicated defined term in parameters. Please check your list of params.")
    params <- append(params, dot.params)
    params <- append(params, list(silent = 1))
    for (mc in metrics) {
        params <- append(params, list(eval_metric = mc))
    }
    if (!is.null(params$objective) && !is.null(obj))
        stop("xgb.cv: cannot assign two different objectives")
    if (!is.null(params$objective))
        if (class(params$objective) == "function") {
            obj = params$objective
            params[["objective"]] = NULL
        }
    if (!is.null(params$eval_metric))
        if (class(params$eval_metric) == "function") {
            feval = params$eval_metric
            params[["eval_metric"]] = NULL
        }
    if (!is.null(early.stop.round)) {
        if (!is.null(feval) && is.null(maximize))
            stop("Please set maximize to note whether the model is maximizing the evaluation or not.")
        if (is.null(maximize) && is.null(params$eval_metric))
            stop("Please set maximize to note whether the model is maximizing the evaluation or not.")
        if (is.null(maximize)) {
            if (params$eval_metric %in% c("rmse", "logloss",
                "error", "merror", "mlogloss")) {
                maximize = FALSE
            }
            else {
                maximize = TRUE
            }
        }
        if (maximize) {
            bestScore = 0
        }
        else {
            bestScore = Inf
        }
        bestInd = 0
        earlyStopflag = FALSE
        if (length(metrics) > 1)
            warning("Only the first metric is used for early stopping process.")
    }
    xgb_folds <- xgb.cv.mknfold(dtrain, nfold, params, stratified,
        folds)
    obj_type = params[["objective"]]
    mat_pred = FALSE
    if (!is.null(obj_type) && obj_type == "multi:softprob") {
        num_class = params[["num_class"]]
        if (is.null(num_class))
            stop("must set num_class to use softmax")
        predictValues <- matrix(0, xgb.numrow(dtrain), num_class)
        mat_pred = TRUE
    }
    else predictValues <- rep(0, xgb.numrow(dtrain))
    history <- c()
    print.every.n = max(as.integer(print.every.n), 1L)
    cvPredList <- list()
    for (i in 1:nrounds) {
        predictValues <- rep(0, xgb.numrow(dtrain))
        msg <- list()
        for (k in 1:nfold) {
            fd <- xgb_folds[[k]]
            succ <- xgb.iter.update(fd$booster, fd$dtrain, i -
                1, obj)
            msg[[k]] <- xgb.iter.eval(fd$booster, fd$watchlist, i - 1, feval) %>% str_split("\t") %>% .[[1]]
            tmpPred <- xgb.iter.eval(fd$booster, fd$watchlist, i - 1, feval, prediction)
            predictValues[fd$index] = tmpPred[[2]]
        }
        cvPredList[[i]] <- predictValues
        ret <- xgb.cv.aggcv(msg, showsd)
        history <- c(history, ret)
        if (verbose)
            if (0 == (i - 1L)%%print.every.n)
                cat(ret, "\n", sep = "")
        if (!is.null(early.stop.round)) {
            score = strsplit(ret, "\\s+")[[1]][1 + length(metrics) +
                2]
            score = strsplit(score, "\\+|:")[[1]][[2]]
            score = as.numeric(score)
            if ((maximize && score > bestScore) || (!maximize &&
                score < bestScore)) {
                bestScore = score
                bestInd = i
            }
            else {
                if (i - bestInd >= early.stop.round) {
                  earlyStopflag = TRUE
                  cat("Stopping. Best iteration:", bestInd)
                  break
                }
            }
        }
    }
    colnames <- str_split(string = history[1], pattern = "\t")[[1]] %>%
        .[2:length(.)] %>% str_extract(".*:") %>% str_replace(":",
        "") %>% str_replace("-", ".")
    colnamesMean <- paste(colnames, "mean")
    if (showsd)
        colnamesStd <- paste(colnames, "std")
    colnames <- c()
    if (showsd)
        for (i in 1:length(colnamesMean)) colnames <- c(colnames,
            colnamesMean[i], colnamesStd[i])
    else colnames <- colnamesMean
    type <- rep(x = "numeric", times = length(colnames))
    dt <- utils::read.table(text = "", colClasses = type, col.names = colnames) %>%
        as.data.table
    split <- str_split(string = history, pattern = "\t")
    for (line in split) dt <- line[2:length(line)] %>% str_extract_all(pattern = "\\d*\\.+\\d*") %>%
        unlist %>% as.numeric %>% as.list %>% {
        rbindlist(list(dt, .), use.names = F, fill = F)
    }
    dt[, test.auc.mean.sd:=test.auc.mean-test.auc.std]
    dt[, tree:=1:.N]
    setorder(dt, -test.auc.mean)
    tree_auc_mean <- dt[1, tree]
    setorder(dt, -test.auc.mean.sd)
    tree_auc_mean_sd <- dt[1, tree]
    setorder(dt, tree)
    new_nrounds <- max(tree_auc_mean_sd, tree_auc_mean)
    dt[, c("test.auc.mean.sd", "tree"):=NULL]
    if (prediction) {
        predictValues <- cvPredList[[new_nrounds]]
        return(list(dt = dt, pred = predictValues, cvPredList = cvPredList))
    }
    return(dt)
}
environment(xgb.cv1) <- environment(xgb.cv)
