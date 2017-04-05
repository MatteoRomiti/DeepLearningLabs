
import time
import random
import numpy as np
from utils import *
from transfer_functions import * 
from pprint import pprint


class NeuralNetwork(object):
    
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, iterations=50, learning_rate = 0.1):
        """
        input: number of input neurons
        hidden: number of hidden neurons
        output: number of output neurons
        iterations: how many iterations
        learning_rate: initial learning rate
        """
       
        # initialize parameters
        self.iterations = iterations   #iterations
        self.learning_rate = learning_rate
     
        
        # initialize arrays
        self.input = input_layer_size+1  # +1 for the bias node in the input Layer
        self.hidden = hidden_layer_size+1 # +1 for the bias node in the hidden layer 
        self.output = output_layer_size

        # set up array of 1s for activations
        self.a_input = np.ones(self.input)
        self.a_hidden = np.ones(self.hidden)
        self.a_output = np.ones(self.output)
        self.o_output = np.ones(self.output)

        #transfer function initialization (from utils.py)
        self.transferFunc = sigmoid
        self.derivateTransferFunc = dsigmoid

        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
       
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers


    def set_transfer_function(self, transferFunc, derivateTransferFunc):
        self.transferFunc = transferFunc
        self.derivateTransferFunc = derivateTransferFunc
       
        
    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs):
        #array of inputs, we append the bias that is 1
        self.a_input = np.append(inputs, [1])

        #we pass the inputs times the weights to the transfer function
        self.a_hidden = np.append(self.transferFunc(self.a_input.dot(self.W_input_to_hidden)), [1])

        #compute what arrives at the output layer
        self.a_output = self.a_hidden.dot(self.W_hidden_to_output)

        #pass this to the transfer function to get the output of the network
        self.o_output = self.transferFunc(self.a_output)
        
        return self.o_output
        

    def backPropagate(self, targets):

        # calculate error terms for output
        self.outputErrors = self.o_output - targets

        #calculate derivative of Error w.r.t. output layer
        dEdu2 = np.multiply(self.outputErrors, self.derivateTransferFunc(self.o_output))
        #calculate derivative of Error w.r.t. hidden layer
        dEdu1 = np.multiply(self.W_hidden_to_output.dot(dEdu2), self.derivateTransferFunc(self.a_hidden))
        dEdu1 = np.delete(dEdu1, -1)

        # update output weights
        self.W_hidden_to_output -= self.learning_rate * np.outer(self.a_hidden, dEdu2)
        # update input weights
        self.W_input_to_hidden -= self.learning_rate * np.outer(self.a_input, dEdu1)
  
        return np.sum(self.outputErrors**2)/2
    
    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
        Validation_accuracies=[]
      
        for it in range(self.iterations):
            np.random.shuffle(data)
            inputs  = [entry[0] for entry in data ]
            targets = [ entry[1] for entry in data ]
            
            error=0.0 
            for i in range(len(inputs)):
                Input = inputs[i]
                Target = targets[i]
                self.feedForward(Input)
                error+=self.backPropagate(Target)
            Training_accuracies.append(self.predict(data))
            Validation_accuracies.append(self.predict(validation_data))
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[=====] -Error: %5.10f  -Training_Accuracy:  %2.2f  -Validation_accuracy: %2.2f -time: %2.2f " \
                %(it+1,\
                self.iterations,\
                error,\
                (self.predict(data)/len(data))*100,\
                (self.predict(validation_data)/len(validation_data))*100,\
                time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        #plot_curve(range(1,self.iterations+1),errors, "Error")
        #plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")

        #returning these two arrays for plotting all together for better visualization
        return (Training_accuracies, errors, Validation_accuracies)
        
     

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count = count 
        return count 
    
    def guessDigit(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            outputVector = self.feedForward(testcase[0])
            prediction = np.argmax(outputVector)
            np.set_printoptions(precision=3, suppress=True)
            print(outputVector)
            print("The model predicts:", prediction, " and the answer is:", answer)
            print("----------------")
            count = count + 1 if (answer - prediction) == 0 else count 
            count = count 
        return count 


    
    
    def save(self, filename):
        """ Save neural network (weights) to a file. """
        with open(filename, 'wb') as f:
            pickle.dump({'wi':self.W_input_to_hidden, 'wo':self.W_hidden_to_output}, f )
        
        
    def load(self, filename):
        """ Load neural network (weights) from a file. """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        # Set biases and weights
        self.W_input_to_hidden=data['wi']
        self.W_hidden_to_output = data['wo']
        
            
                                  
                                  
    
  



    
    
   