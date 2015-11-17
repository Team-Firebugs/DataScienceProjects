
## This function is about creating a list containing 4
## functions that set the matrix x, get the matrix x,
## set the inverse of matrix x, and get this inverse
## matrix.
## 1. Defining m as NULL
## 2. Defining a function that sets y, and set it into x, in
##    the cache. Reset m to NULL in the cache.
## 3. Setting the function that returns the matrix x
## 4. Defining the function that sets the inverse of the
##    matrix and store it in the cache as m
## 5. Returning the inverse of the matrix
## 6. Returning the special vector containing the 4
##    functions defined previously.

makeCacheMatrix <- function(x = matrix()) {
    m <- NULL                                       #1
    
    set <- function(y){                             #2
        x <<- y
        m <<- NULL
    }
    
    get <- function() x                             #3
    
    setinverse <- function(solve) m <<- solve       #4
    
    getinverse <- function() m                      #5
    
    list(set = set, get = get,                      #6
         setinverse = setinverse,
         getinverse = getinverse)
}

##--------------------------------------------------------

## This function is made to collect the special vector
## previously created with the makeCacheMatrix function and
## to return its inverse.
## 1. Using the getinverse() function from x to store the
##    inverse matrix into m.
## 2. If m is not NULL, return the message and m, the
##    inverse of the matrix.
## 3. IF the previous condition is not reached, meaning
##    m is NULL, we get here the matrix from the function
##    stored in the special vector and store this matrix
##    into "data".
## 4. We use the function solve to calculate the inverse
##    of the matrix stored in "data" and the store the
##    result into m.
## 5. Setting the inverse of the matrix in the cache
## 6. Returning m

cacheSolve <- function(x, ...) {
        ## Return a matrix that is the inverse of 'x'
    m <- x$getinverse()                             #1
    
    if(!is.null(m)){                                #2
        message("getting cached data")
        return(m)
    }
    
    data <- x$get()                                 #3
    
    m <- solve(data, ...)                           #4
    
    x$setinverse(m)                                 #5
    
    m                                               #6
}
