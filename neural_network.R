# Setup R -----------------------------------------------------------------

# Basically we could do it all with base R, but dplyr is a bit more convenient
library(dplyr)

# Seed to make it replicable
set.seed(123)

# All custom functions are created in here
source("functions.R")


# Data --------------------------------------------------------------------

# Load the MNIST data
d <- readr::read_csv("train.csv", n_max = 1e4)

# Split of the label, spread label factor levels to binary variables (model matrix format) and transpose to make the matrix multiplication easier
# Rows are the labels (0 to 9) and columsn are the observations
y <- model.matrix(~as.factor(d$label)-1) %>%  t()

# Remove the label from the df, gives you a df with all covariates (pixels)
d <- d %>% dplyr::select(-label) %>% as.matrix() 

# Split randomly into train and test set by 2/3 to 1/3
s <- sample(1:nrow(d), size = (nrow(d)*2/3), replace = FALSE)
dTrain <- d[s,] # train pixels
dTest <- d[-s,] # test pixels
yTrain <- y[,s] # train labels in transposed model matrix form
yTest <- y[,-s] # test labels in transposed model matrix form

rm(d, y) ; gc() # clean up


# Setup network -----------------------------------------------------------

# Initialize your network with random weights
# Input are the number of input nodes (784 in the MNIST case, the number of pixels)
# nLayer1 are the number of nodes in the first layer
# nLayer2 are the number of nodes in the second layer
# nOutput are the number of output nodes (10 in the MNIST case if you select all numbers)
initNetwork(nInput = ncol(dTrain), nLayer1 = 10, nLayer2 = 10, nOutput = ifelse(is.null(ncol(yTrain)), 1, nrow(yTrain)))


# Setup backpropagation ---------------------------------------------------

# Determine how aggressive your algorithm is, we use the same for weigth and bias
aggressiveness <- 0.1 
noise <- 0.01

# Number of training steps
iters <- 1:1e3 

# Lists to store intermediary results
trainCost <- list() 
testCost <- list()

# Start your backpropagation
for (i in iters) {
  
  # We calculate the derivates and substract them from the current weights and biases
  W3 <- W3 - aggressiveness * JW3(x = dTrain, y = yTrain) * (1 + rnorm(n = 1) * noise)
  W2 <- W2 - aggressiveness * JW2(x = dTrain, y = yTrain) * (1 + rnorm(n = 1) * noise)
  W1 <- W1 - aggressiveness * JW1(x = dTrain, y = yTrain) * (1 + rnorm(n = 1) * noise)
  
  b3 <- b3 - aggressiveness * Jb3(x = dTrain, y = yTrain) * (1 + rnorm(n = 1) * noise)
  b2 <- b2 - aggressiveness * Jb2(x = dTrain, y = yTrain) * (1 + rnorm(n = 1) * noise)
  b1 <- b1 - aggressiveness * Jb1(x = dTrain, y = yTrain) * (1 + rnorm(n = 1) * noise)
  
  # We store both the trainig and the test cost at every iteration and print it
  trainCost[i] <- cost(dTrain, yTrain)
  testCost[i] <- cost(dTest, yTest)
  print( paste0(i, "    ", trainCost[i], "   ", testCost[i]) )
  
  # Every 10th iteration, we print the train and test cost as well as the table of real and predicted outputs for both the train and the test data set
  # As soon as all labels are present in the prediction, we also calculate the number of correctly classified cases via the dioganal
  if(i %in% seq(from = 1, to = max(iters), by = 100)){
    
    # Train table
    feedForward(dTrain)
    trainResult <- apply(X = a3, MARGIN = 2, FUN = which.max)
    actualResult <- apply(X = yTrain, MARGIN = 2, FUN = which.max)
    print(table(trainResult, actualResult))
    
    # Test table
    feedForward(dTest)
    testResult <- apply(X = a3, MARGIN = 2, FUN = which.max)
    actualResult <- apply(X = yTest, MARGIN = 2, FUN = which.max)
    print(table(testResult, actualResult))
    if(nrow(table(testResult, actualResult)) == 10 ) print(sum(diag(table(testResult, actualResult)))/ncol(yTest))
    
  }
  

}

# Plot the results
data.frame(i = 1:length(trainCost), trainCost = unlist(trainCost), testCost = unlist(testCost)) %>% 
  tidyr::gather(key = key, value = value, 2:3) %>% 
  ggplot(aes(x = i, y = value, fill = key)) + geom_line()

# Final table for train data
feedForward(dTrain)
predictionTrain <- apply(X = a3, MARGIN = 2, FUN = which.max)
actualResult <- apply(X = yTrain, MARGIN = 2, FUN = which.max)
print(table(predictionTrain, actualResult))

# Final table for test data
feedForward(dTest)
predictionTest <- apply(X = a3, MARGIN = 2, FUN = which.max)
actualResult <- apply(X = yTest, MARGIN = 2, FUN = which.max)
print(table(predictionTest, actualResult))
