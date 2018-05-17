
# Activation function sigma(z) is the logistic function
sigma <- function(z) {
  1 / (1 + exp(-z))
}

# Derivative of the activation function derivative_sigma(z)
derivative_sigma <- function(z) {
  ( cosh(z/2)^(-2) ) / 4
}

# Initialize your network, 3 layer of weights and biases
initNetwork <- function(nInput, nLayer1, nLayer2, nOutput){
  
  W1 <<- matrix(rnorm(nInput * nLayer1), ncol = nInput) / 2
  W2 <<- matrix(rnorm(nLayer1 * nLayer2), ncol = nLayer1) / 2
  W3 <<- matrix(rnorm(nLayer2 * nOutput), ncol = nLayer2) / 2
  
  b1 <<- rnorm(nLayer1) / 2
  b2 <<- rnorm(nLayer2) / 2
  b3 <<- rnorm(nOutput) / 2
  
}

# Feed forward and input through the whole network and to see which output nodes are activated
# Equations are
# a^n = sigma(z^n)
# z^n = W^n * a^n-2 + b^n
feedForward <- function(input){
  
  z1 <<- W1 %*% t(input) + b1
  a1 <<- sigma(z1)
  z2 <<- W2 %*% a1 + b2
  a2 <<- sigma(z2)
  z3 <<- W3 %*% a2 + b3
  a3 <<- sigma(z3)
  
}

# Average Cost Function over the Training set 1/N * sum(C)
# where cost is the RSS, so (yHat - y)^2 
cost <- function(x, y){
  
  rss <- sum((feedForward(x) - y)^2) / ncol(y)
  return(rss)
  
}

# Jacobian for the third layer weights
# ^n indicate the layers in the following
# dC/dW3 is by use of chain rule
# dC/dW3 = dC/da^3 * da^3/dz^3 * dz^3/dW^3
# where dC/da^3 = 2 * (a^3 - y)
# where da^3/dz^3 = derivate_sigma( z^3 )
# where dz^2/dW^3 = a^2
JW3 <- function(x, y){
  
  feedForward(x) # puts result in global env
  
  J = 2 * (a3 - y)
  J = J * derivative_sigma(z3)
  J = J %*% t(a2) / ncol(x)
  
  return(J)
  
}

# Jabion for the third layer bias
# ^n indicate the layers in the following
# dC/db3 is by use of chain rule
# dC/db3 = dC/da^3 * da^3/dz^3 * dz^3/db^3
# where dC/da^3 = 2 * (a^3 - y)
# where da^3/dz^3 = derivate_sigma( z^3 )
# where dz^2/db^3 = 1
Jb3 <- function(x, y){
  
  feedForward(x)
  
  J = 2 * (a3 - y)
  J = J * derivative_sigma(z3)
  
  J = rowSums(J) / ncol(x)
  
  return(J)
  
}

# Jacobian for the second layer weights are similar but with an identical partial derivate term
# ^n indicate the layers in the following
# dC/dW2 is by use of chain rule
# dC/dW2 = dC/da^3 * da^3/da^2 * da^2/dz^2 * dz^2/dW^2
# where the new da^3/da^2 = da^3/dz^3 * dz^3/da^2 = derivative_sigma( z^3 ) * W^3
JW2 <- function(x, y){
  
  feedForward(x) # puts result in global env
  
  J = 2 * (a3 - y)
  J = J * derivative_sigma(z3)
  J = t(t(J) %*% W3)
  J = J * derivative_sigma(z2)
  
  J = J %*% t(a1) / ncol(x)
  
  return(J)
  
}

# Jabion for the second layer bias
# and so on
Jb2 <- function(x, y){
  
  feedForward(x)
  
  J = 2 * (a3 - y)
  J = J * derivative_sigma(z3)
  J = t(t(J) %*% W3)
  J = J * derivative_sigma(z2)
  
  J = rowSums(J) / ncol(x)
  
  return(J)
  
}

# Jacobian for the first layer weights
# and so on
JW1 <- function(x, y){
  
  feedForward(x) # puts result in global env
  
  J = 2 * (a3 - y)
  J = J * derivative_sigma(z3)
  J = t(t(J) %*% W3)
  J = J * derivative_sigma(z2)
  J = t(t(J) %*% W2)
  J = J * derivative_sigma(z1)
  
  J = J %*% x / ncol(x)
  
  return(J)
  
}

# Jabion for the first layer bias
Jb1 <- function(x, y){
  
  feedForward(x)
  
  J = 2 * (a3 - y)
  J = J * derivative_sigma(z3)
  J = t(t(J) %*% W3)
  J = J * derivative_sigma(z2)
  J = t(t(J) %*% W2)
  J = J * derivative_sigma(z1)
  
  J = rowSums(J) / ncol(x)
  
  return(J)
  
}
