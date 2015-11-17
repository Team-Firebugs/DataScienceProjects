pollutantmean <- function(directory, pollutant, id = 1:332){
    z <- c()
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
        
        col_pol <- f[,pollutant]
        z <- c(z,col_pol[!is.na(col_pol)])
    }
    mean(z)
}