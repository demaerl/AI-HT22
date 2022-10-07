# Group 3
# TODO: names

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
  DIM = 40
  
  # 1: determine graph
  # only in first game and first iteration
  if(moveInfo$mem$status == 0 && is.null(moveInfo$mem$trans)) {
    l <- getTransition(edges, DIM)
    adj <- l$adj
    Ts <- l$trans
    moveInfo$mem$trans <- Ts
    moveInfo$mem$adj <- adj
  } else {
    adj <- moveInfo$mem$adj
    Ts <- moveInfo$mem$trans
  }
  
  # 2: calc state matrix
  if(moveInfo$mem$status == 0 | moveInfo$mem$status == 1) {
    S <- getInitialState(positions, DIM)
  } else {
    S <- moveInfo$mem$state
  }
  
  # 3: emission calculations
  E <- getEmission(probs, readings, DIM)
  # 4: matrix calculations
  S <- getNextState(S, Ts, E)
  S <- getStateWithPositions(S, positions)
  
  # 5: determine target node (max prob)
  max_p_node <- which(S == max(S), arr.ind = TRUE)
  target_pos <- max_p_node[2]
  
  curr_pos <- positions[3]
  
  # 6: shortest path
  path <- shortestPath(curr_pos, target_pos, adj, DIM)
  move <- getNextMove(path)
  
  # update prob of searched nodes
  if(move[1] == 0) {
    S[curr_pos] <- 0
  }
  else if(move[2] == 0) {
    next_pos <- move[1]
    S[next_pos] <- 0
  }
  moveInfo$mem$state <- S
  
  moveInfo$moves <- move
  #browser()
  
  moveInfo$mem$status <- 2
  return(moveInfo)
}

getInitialState <- function(pos, dim) {
  S <- matrix(0, nrow = 1, ncol = 40)
  t1 <- pos[1]
  t2 <- pos[2]
  
  if(is.na(t1)) {t1 <- 0}
  if(is.na(t2)) {t2 <- 0}

  # both tourists eaten
  if(t1 < 0 && t2 < 0) {
    S[abs(t1)] <- 1 
  }
  else if(t1 < 0) {
    S[abs(t1)] <- 1 
  }
  else if(t2 < 0) {
    S[abs(t2)] <- 1 
  }
  else if(t1 != 0 && t2 != 0) {
    S <- matrix(1/(40 -2), nrow = 1, ncol = 40)
    S[t1] <- 0
    S[t2] <- 0
  }
  return (S)
}

getStateWithPositions <- function(S, pos) {
  S_new <- S
  t1 <- pos[1]
  t2 <- pos[2]
  
  if(is.na(t1)) {t1 <- 0}
  if(is.na(t2)) {t2 <- 0}
  
  if(t1 < 0 && t2 < 0) {
    S_new <- matrix(0, nrow = 1, ncol = 40)
    S_new[abs(t1)] <- 1

  }
  else if(!is.na(t1) && t1 < 0) {
    S_new <- matrix(0, nrow = 1, ncol = 40)
    S_new[abs(t1)] <- 1
    
  }
  else if(!is.na(t2) && t2 < 0) {
    S_new <- matrix(0, nrow = 1, ncol = 40)
    S_new[abs(t2)] <- 1
  }
  return (S_new)
}

getNextState <- function(S, T, E) {
  tmp <- S %*% T
  S_new <- tmp*t(E)
  S_new <- S_new/sum(S_new)
  return (S_new)
}

getNextMove <- function(path, currPos) {

  if(length(path) >= 2) {
    return (c(path[1], path[2]))
  }
  else if(length(path) == 1) {
    return (c(path[1], 0))
  }
  else
    return (c(0, 0))
}

getTransition <- function(edges, dim) {
  adj <- matrix(0, nrow = 40, ncol = 40)
  trans <- adj

  for(i in 1:nrow(edges)) {
    e <- edges[i,]
    start <- e[1]
    end <- e[2]
    adj[start, end] <- 1
    adj[start, start] <- 1
    adj[end, start] <- 1
  }
  
  for(i in 1:40) {
    row <- adj[i, ]
    sum <- sum(row)
    if(sum != 0) {
      trans[i, ] <- adj[i,] / sum
    }
  }
  l <- list("adj" = adj, "trans" = trans)
  return(l)
}

getEmission <- function(probs, read, dim) {
  emissions <- matrix(0, nrow = 40, ncol = 1)
  
  for(i in 1:40) {
    sal <- probs$salinity[i,]
    phos <- probs$phosphate[i,]
    nit <- probs$nitrogen[i,]
    
    sal_p <- dnorm(read[1], sal[1], sal[2])
    phos_p <- dnorm(read[2], phos[1], phos[2])
    nit_p <- dnorm(read[3], nit[1], nit[2])
    
    p_node <- sal_p * phos_p * nit_p
    emissions[i] <- p_node
  }
  return(emissions)
}

shortestPath = function(start, goal, adj, dim) {
  #BFS seach
  visited = c(start)
  frontier = c(start)
  parents = replicate(40, 0)
  parents[start] = -1
  while (length(frontier) != 0) {
    current = head(frontier, n=1)
    frontier = setdiff(frontier, c(current))
    row = adj[current,]
    neighbors = c(which(row == 1))
    neighbors = setdiff(neighbors, c(current))
    neighbors = setdiff(neighbors, visited)
    for (node in neighbors) {
      if (!(node %in% visited)) {
        frontier = c(frontier, node)
        parents[node] = current
        visited = c(visited, c(node))
      }
    }
  }
  
  current = goal
  path = numeric()
  while (current != -1) {
    if (parents[current] != -1) {
      path = c(c(current), path)
    }
    current = parents[current]
  }
  
  return (path)
}

# Of course you can have supporting functions outside to delegate repetitive 
# tasks or to make your code more readable.