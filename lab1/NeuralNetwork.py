import time
import random
import numpy as np
from utils import *
from transfer_functions import * 


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
        self.a_input = np.zeros(self.input)
        self.a_hidden = np.zeros(self.hidden)
        self.a_output = np.zeros(self.output)
        
        
        #create randomized weights Yann Lecun method in 1988's paper ( Default values)
        input_range = 1.0 / self.input ** (1/2)
        self.W_input_to_hidden = np.random.normal(loc = 0, scale = input_range, size =(self.input, self.hidden-1))
        self.W_hidden_to_output = np.random.uniform(size = (self.hidden, self.output)) / np.sqrt(self.hidden)
       
        
    def weights_initialisation(self,wi,wo):
        self.W_input_to_hidden=wi # weights between input and hidden layers
        self.W_hidden_to_output=wo # weights between hidden and output layers
   
        
       
        
    #========================Begin implementation section 1============================================="    
    
    def feedForward(self, inputs):
        
        # Compute input activations
        if len(inputs) < self.input:
            inputs.append(1)
        self.a_input = np.asarray(inputs)
        # Compute  hidden activations
        self.a_hidden = np.dot(self.W_input_to_hidden.transpose(),self.a_input)
        # Compute output activations
        if self.a_hidden.shape[0] < self.hidden:
            self.a_hidden = np.append(self.a_hidden, 1)
        self.a_output = np.dot(self.W_hidden_to_output.transpose(),sigmoid(self.a_hidden))
        #       for i in range(self.output): # i index for neuron in output layer
#               for j in range(self.hidden): # j index for neuron in hidden layer
#                   self.a_output[i] = + self.W_hidden_to_output[j,i]*sigmoid(self.a_hidden[j])
#        out = self.a_output
#        for i in range(self.output):
#                out[i] = sigmoid(self.a_output[i])
#        return out
 

       
     #========================End implementation section 1==============================================="   
        
        
        
        
     #========================Begin implementation section 2=============================================#    

    #def backPropagate(self, targets):
        
        
         # calculate error terms for output
       
        # calculate error terms for hidden
        
        # update output weights
      
        # update input weights
        
        # calculate error
        
        
     #========================End implementation section 2 =================================================="   

    
    
    
    def train(self, data,validation_data):
        start_time = time.time()
        errors=[]
        Training_accuracies=[]
      
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
            
            error=error/len(data)
            errors.append(error)
            
           
            print("Iteration: %2d/%2d[==============] -Error: %5.10f  -Training_Accuracy:  %2.2f  -time: %2.2f " %(it+1,self.iterations, error, (self.predict(data)/len(data))*100, time.time() - start_time))
            # you can add test_accuracy and validation accuracy for visualisation 
            
        plot_curve(range(1,self.iterations+1),errors, "Error")
        plot_curve(range(1,self.iterations+1), Training_accuracies, "Training_Accuracy")
       
        
     

    def predict(self, test_data):
        """ Evaluate performance by counting how many examples in test_data are correctly 
            evaluated. """
        count = 0.0
        for testcase in test_data:
            answer = np.argmax( testcase[1] )
            prediction = np.argmax( self.feedForward( testcase[0] ) )
            count = count + 1 if (answer - prediction) == 0 else count 
            count= count 
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
        
            
                                  
                                  
    
  



    
    
   