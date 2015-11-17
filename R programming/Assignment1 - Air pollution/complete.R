complete <- function(directory, id = 1:332){
    fin_id <- c()
    fin_cc <- c()
    
    for (i in id){
        if (nchar(as.character(i))==1){
            new_id <- paste("00", as.character(i), sep="")
        }
        else if (nchar(as.character(i))==2){
            new_id <- paste("0", as.character(i), sep="")
        }
        else if (nchar(as.character(i))==3){
            new_id <- as.character(i)
        }
        
        new_dir <- paste(directory, "/", new_id, ".csv", sep="") 
        f <- read.csv(new_dir)
        
        good <- complete.cases(f)
        fin_id <- c(id)
        fin_cc <- c(fin_cc,nrow(f[good,]))
    }
    x <- data.frame(id = fin_id, nobs = fin_cc)
    x
}