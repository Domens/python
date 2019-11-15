import pandas as pd
import numpy as np
from random import seed
from random import random
from math import e
import sys

class NN:
    def __init__(self, form_of_neural,df, learning_rate = 0.1):
        self.train_arr = df[df.index<len(df)*0.7].to_numpy()
        self.test_arr= df[df.index>=len(df)*0.7].to_numpy()
        self.df = df
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append(Layer(form_of_neural[0],form_of_neural[0], form_of_neural[1]))
        for i in range(1,len(form_of_neural)-1):
            self.layers.append(Layer(form_of_neural[i],form_of_neural[i-1], form_of_neural[i+1]))
        self.layers.append(Layer(form_of_neural[-1],form_of_neural[-2], 0))
        
    def forward_propagation(self, start_inp_arr):
        out = []
        out.append(self.layers[0].forward(start_inp_arr))
        for i in range(1,len(self.layers)):
            out.append(self.layers[i].forward((out[-1])))
        return out[-1]
    
    def backward_propagation(self, goal_arr, start_inp_arr):
        sigms = []
        out_neurons = self.layers[-1].neurons
        for i in range(len(goal_arr)):
            sigms.append(out_neurons[i].out*(1- out_neurons[i].out)*(goal_arr[i]- out_neurons[i].out))
        
        for i in reversed(range(len(self.layers))):
            if i !=0:
                sigms = self.layers[i].backward(sigms, self.layers[i-1].give_outputs(), self.learning_rate)
            else:
                self.layers[i].backward(sigms, start_inp_arr, self.learning_rate)

    def train(self, n_epoch):
        for n in range(n_epoch):
            for i in range(len(self.train_arr)):
                self.forward_propagation(self.train_arr[i][:len(self.train_arr[0])-1])
                self.backward_propagation(self.goal_generate(self.train_arr[i][len(self.train_arr[0])-1]), self.train_arr[i][:len(self.train_arr[0])-1])
               

            T = 0
            for i in range(len(self.test_arr)):
                if (np.argmax(self.forward_propagation(self.test_arr[i][:len(self.test_arr[0])-1]))+1) == (self.test_arr[i][len(self.test_arr[0])-1]):
                    T+=1
            print("accuracy" )
            print(T/len(self.test_arr))
        
            
    def goal_generate(self,col):
        
        result = []
        for i in range(len(self.df.iloc[:,-1].unique())):
            result.append(0)
        result[int(col)-1] = 1
        return result
            
class Layer:
    def __init__(self, num_of_neurons, num_of_input, num_of_outputs):
        self.neurons = []
        for i in range(num_of_neurons):
            self.neurons.append(Neuron(num_of_input, num_of_outputs))
    
    def forward(self, inp_arr):
        out = []
        for n in self.neurons:
            out.append(n.activate(inp_arr))
        return out
    
    def backward(self,sigms_arr, inp_arr,learning_rate):
        for n in range(len(self.neurons)):
            self.neurons[n].sigma = sigms_arr[n]
        
        for n in self.neurons:
            n.actualize_weights(inp_arr,learning_rate)
        
        temp = []
        for n in self.neurons:
            temp.append(n.new_sigms())
        
        new_sigms_arr = []
       
        for i in range(len(temp[0])):
            new_sigms_arr.append(0)
            for idx in range(len(temp)):
                new_sigms_arr[i]+=temp[idx][i]

        return new_sigms_arr
    def give_outputs(self):
        out = []
        for n in self.neurons:
            out.append(n.out)
        return out
    
    
class Neuron:
    def __init__(self, num_of_input, num_of_outputs):
        #self.weights = [random() for i in range( num_of_input)]
        self.weights = np.random.normal(
                loc = 0.0,
                scale = np.sqrt(2 / (num_of_input + num_of_outputs)),
                size = ( num_of_input))
       
        self.bias = 0
        self.out = 0
        self.sigma = 0
    
    def activate(self, inp_arr):
        self.out = self.bias
        for i in range(len(self.weights)):
            self.out += inp_arr[i] * self.weights[i]
        self.out = self.sigmoid(self.out)
        return self.out
        
    def sigmoid(self, num):
        return 1/(1+e**-num)

    def actualize_weights(self,inp_arr,learning_rate):
        for i in range(len(self.weights)):            
            self.weights[i]+=self.sigma * inp_arr[i]*learning_rate


    def new_sigms(self):
        sigms = []
        for i in range(len(self.weights)):
            sigms.append(self.sigma*self.weights[i]*(1-self.out)*self.out)
        return sigms
            
layer_structure = [13, 20, 30, 3] #first = num of inputs, last = num_of_outputs
learning_rate = 0.1
path = "wine.csv"
some = []
df = pd.read_csv(path)
df = df[reversed(list(df.columns))]
df = df.sample(frac=1).reset_index(drop=True)
temp = df['Wine']
df=(df-df.mean())/df.std()
df['Wine'] = temp
neural  = NN(layer_structure,df,learning_rate)
neural.train(50)
