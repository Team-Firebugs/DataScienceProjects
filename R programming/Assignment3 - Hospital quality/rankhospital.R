rankhospital <- function(state, outcome, num = "best"){
    ## Read outcome data
    f <- read.csv("outcome-of-care-measures.csv", colClasses = "character")
    
    ## Check that state and outcome are valid
    if (!(state %in% f[,"State"])){
        stop("Invalid state")
    }
    v <- c("heart attack","heart failure","pneumonia")
    if (!(outcome %in% v)){
        stop("Invalid outcome")
    }
    
    ## Return hospital name in that state with the given rank
    ## 30-day death rate
    if (outcome == "heart attack"){oc <- 11}
    if (outcome == "heart failure"){oc <- 17}
    if (outcome == "pneumonia"){oc <- 23}
    
    a <- split(f,f[,"State"])
    dd <- data.frame(a[[state]][,"Hospital.Name"], as.numeric(a[[state]][,oc]))
    group <- complete.cases(dd)
    dd2 <- dd[group,]
    dd3 <- dd2[order(dd2[,2],dd2[,1]),]
    
    if (num == "best"){num <- 1}
    if (num == "worst"){num <- nrow(dd3)}
    if (num <= nrow(dd3)){t <- as.character(dd3[num,1])}
    if (num > nrow(dd3)){t <- NA}
    t
}
   
