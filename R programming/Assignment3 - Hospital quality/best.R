best <- function(state, outcome) {
    #Read the file
    f <- read.csv("outcome-of-care-measures.csv", colClasses = "character")
    
    #Check State et outcome
    if (!(state %in% f[,"State"])){
        stop("Invalid state")
    }
    v <- c("heart attack","heart failure","pneumonia")
    if (!(outcome %in% v)){
        stop("Invalid outcome")
    }
    
    if (outcome == "heart attack"){oc <- 11}
    if (outcome == "heart failure"){oc <- 17}
    if (outcome == "pneumonia"){oc <- 23}
    
    a <- split(f,f[,"State"])
    e <- as.numeric(a[[state]][,oc])
    f <- e[!is.na(e)]
    b <- min(f)
    
    vect <- c()
    for (i in seq(1,length(e))){
        c <- e[i]
        if (!is.na(c)){
            if (c == b){
                vect <- c(vect,a[[state]][i,"Hospital.Name"])
            }
        }
    }
    
    vect <- sort(vect)
    
    vect[1]
    
    #heart attack 11
    #heart failure 17
    #pneumonia 23
    
    #Hospital.Name
    
    #State
}