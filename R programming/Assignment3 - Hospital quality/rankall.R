rankall <- function(outcome, num = "best"){
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
    
    ## For each state, find the hospital of the given rank
    if (outcome == "heart attack"){oc <- 11}
    if (outcome == "heart failure"){oc <- 17}
    if (outcome == "pneumonia"){oc <- 23}
    
    a <- split(f,f[,"State"])
    hosp <- c()
    st <- c()
    t <- num
    
    for (each in names(a)){
    num <- t
    dd <- data.frame(a[[each]][,"Hospital.Name"], as.numeric(a[[each]][,oc]))
    group <- complete.cases(dd)
    dd2 <- dd[group,]
    dd3 <- dd2[order(dd2[,2],dd2[,1]),]
    
    if (num == "best"){num <- 1}
    if (num == "worst"){num <- nrow(dd3)}
    if (num <= nrow(dd3)){hosp <- c(hosp,as.character(dd3[num,1]))}
    if (num > nrow(dd3)){hosp <- c(hosp,NA)}
    
    st <- c(st,each)
    
    }
    ## Return a data frame with the hospital names and the
    ## (abbreviated) state name
    data <- data.frame("hospital" = hosp, "state" = st)
    data2 <- data[order(data[,2], data[,1]),]
    
    data2

}
    
