corr <- function(directory, threshold = 0) {
    v <-  c()
    a <- complete(directory,1:332)
    for (i in 1:nrow(a)){
        if (a[i,2]>threshold){
            id_test <- a[i,1]
            if (nchar(as.character(a[i,1]))==1){
                new_id <- paste("00", as.character(a[i,1]), sep="")
            }
            else if (nchar(as.character(a[i,1]))==2){
                new_id <- paste("0", as.character(a[i,1]), sep="")
            }
            else if (nchar(as.character(a[i,1]))==3){
                new_id <- as.character(a[i,1])
            }
        new_dir <- paste(directory, "/", new_id, ".csv", sep="") 
        f <- read.csv(new_dir)
        
        good <- complete.cases(f)
        b <- f[good,]
            
        v <- c(v,cor(b[,"sulfate"],b[,"nitrate"]))
        }
    }
    v
}