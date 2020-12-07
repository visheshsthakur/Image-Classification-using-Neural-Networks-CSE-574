import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return(1.0/ (1.0 + np.exp(-z))) # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
   # loads the MAT object as a Dictionary
            
        #print(max(mat["train0"][1]))

        # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
        # Your code here.
    train_data_pre = np.zeros(shape=(50000, len(mat["train0"][1])))
    validation_data_pre = np.zeros(shape=(10000, len(mat["train0"][1])))
    test_data_pre = np.zeros(shape=(10000, len(mat["train0"][1])))
    train_label_pre = np.zeros(shape=(50000,))
    validation_label_pre = np.zeros(shape=(10000,))
    test_label_pre = np.zeros(shape=(10000,))
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0

    for i in mat:
        if "train" in  i:
            value = mat[i]
            d = len(value)
            leng = d - 1000
            train_data_pre[train_len:train_len+leng] = value[1000:]
            train_len += leng
            #print(train_len)

            train_label_pre[train_label_len:train_label_len + leng] = i[-1]
            #print(train_label_pre)
            train_label_len += leng

            validation_data_pre[validation_len:validation_len + 1000] = value[0:1000]
            validation_len += 1000

            validation_label_pre[validation_label_len:validation_label_len + 1000] = i[-1]
            #print(validation_label_pre)
            validation_label_len += 1000
        if "test" in i:
            value = mat[i]
            d = len(value)
            test_label_pre[test_len:test_len + d] = i[-1]
            test_data_pre[test_len:test_len + d] = value
            #print(test_data_pre)
            test_len += d

   ######################### Shuffle and Normalize###############################################

    train_size = len(train_data_pre)
    train_perm = np.random.permutation(train_size)
    train_data = train_data_pre[train_perm]
    train_data = (train_data - min(mat["train0"][1])) / (max(mat["train0"][1]) - min(mat["train0"][1])) ##Normalize
    train_data = np.double(train_data)
    train_label = train_label_pre[train_perm]

    validation_size = len(validation_data_pre)
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_data_pre[vali_perm]
    validation_data = (validation_data - min(mat["train0"][1])) / (max(mat["train0"][1]) - min(mat["train0"][1]))
    validation_data = np.double(validation_data)
    validation_label = validation_label_pre[vali_perm]

    test_size = len(validation_data_pre)
    test_perm = np.random.permutation(test_size)
    test_data = test_data_pre[test_perm]
    test_data = (test_data - min(mat["train0"][1])) / (max(mat["train0"][1]) - min(mat["train0"][1]))
    test_data = np.double(test_data)
    test_label = test_label_pre[test_perm]

    features_to_delete = []



    #print(train_data.shape)        



    ####### Deleting uselesss features ##########################
    for i in range(len(mat["train0"][1])):
        # If feature is of no importance in training data
        if max(train_data[:,i]) - min(train_data[:,i])  == 0:
            # if feature is of no importance in validation data
            if max(validation_data[:,i]) - min(validation_data[:,i]) == 0:
                # same check on test data
                if max(test_data[:,i]) - min(test_data[:,i]) == 0:
                    #print(1)
                    features_to_delete.append(i)
        else:
            selected_feature.append(i)



    train_data = np.delete(train_data, features_to_delete, axis=1)
    validation_data = np.delete(validation_data, features_to_delete, axis=1)
    test_data = np.delete(test_data, features_to_delete, axis=1)


    #print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label





def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Add Bias
    training_data = np.column_stack((training_data,np.ones(train_data.shape[0])))

    ####### Forward Pass



    ######### Hidden Layer
    hiddenOutput = np.dot(training_data, w1.T)
    hiddenOutput = sigmoid(hiddenOutput)
    # Bias
    hiddenOutput = np.column_stack((hiddenOutput,np.ones(train_data.shape[0])))  #add column  
    ##### Output Layer

    Finaloutput = np.dot(hiddenOutput, w2.T)

    Finaloutput=sigmoid(Finaloutput)

    Class = np.zeros((training_data.shape[0],n_class)) 


    for i in range(training_label.shape[0]):
        position = int(training_label[i])
        Class[i][position] = 1

    # Starting the backward pass
    size = training_data.shape[0]
    ##################### Lecture 9 ################################
    ###Error Function
    J = ((Class * np.log(Finaloutput)) + ((1 - Class) * np.log((1 - Finaloutput))))
    error_func = ((-1) * np.sum(J))/size



    ####### Regularization

    Reg = (np.sum(w1**2) + np.sum(w2**2)) * (lambdaval/(2*size))

    obj_val = error_func + Reg

    #################Gradient####################
    ########## Lecture 9 Handouts############################
    delta = Finaloutput - Class

    grad_w2 = np.dot(((delta).T), hiddenOutput) 

    grad_w2 = (grad_w2+(lambdaval*w2))/size



    grad_w1 = np.dot(((1-hiddenOutput)*hiddenOutput* (np.dot(delta,w2))).T,training_data)
    grad_w1 = np.delete(grad_w1, n_hidden, axis=0)
    grad_w1 = (grad_w1 + (lambdaval * w1)) /size
    #print(grad_w1)

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    # Number of Items  
    labels = np.array([])
    # Your code here
    data = np.column_stack((data,np.ones(data.shape[0])))
    # Ïƒ(w^t * x )
    hidden_layer_output = sigmoid(np.dot(data, w1.T))
    hidden_layer_output = np.column_stack((hidden_layer_output, np.ones(hidden_layer_output.shape[0])))
    output_layer_output = sigmoid(np.dot(hidden_layer_output, w2.T))
    labels = np.argmax(output_layer_output, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""
selected_feature = []
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 10

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
pickle.dump((selected_feature,n_hidden,w1,w2,lambdaval),open('params.pickle','wb'))
