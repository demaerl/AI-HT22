# Add your group details ...

#' myFunction
#' 
#' You can test the outcome of myFunction using the following lines of code. It 
#' will print out the results generated against the test function. Note that the
#' results will vary depending on the solution and approach. Please refer to the
#' Passing Requirements at the Studium assignment page.
#' 
#' > library(WheresCroc)
#' > testWC(myFunction, verbose = 1)
#' 
#' Mean moves: 5.444
#' SD moves: 3.853
#' Time taken: 6.5 seconds.[1] 5.444
#' 
#' -----------------------------------------------------------------------------
#' Available parameters to myFunction. 
#' (More details at runWheresCroc documentation in WheresCroc package)
#' 
#' @param moveInfo = list(moves = c(move1, move2), mem)
#' @param readings = c(salinity, phosphate, nitrogen)
#' @param positions = c(pos1, pos2, pos3)
#' @param edges = matrix(ncol=2)
#' @param probs = list(mat1, mat2, mat3)
#' 
#' @return See runWheresCroc for details
#' 
#' @export
myFunction <- function(moveInfo, readings, positions, edges, probs) {
  # Start from here...
  moveInfo <- list(moves = c(1, 0), NULL)
  print(readings)
  print(positions)
  print(edges)
  print(probs)
  # End here...
  return(moveInfo)
}

runWheresCroc(myFunction, doPlot = T, showCroc = F, pause = 1, verbose = T, returnMem = F, mem = NA)

# Of course you can have supporting functions outside to delegate repetitive 
# tasks or to make your code more readable.