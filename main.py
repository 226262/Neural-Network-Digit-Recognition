#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy
from  neural_network import Neural_Network
from plotter import input_stuff


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


#writing a circle of pixels to the array according to middle coordinates of mouse position

NN=Neural_Network()

if input("Wanna train model? y/N ") =='y':
    layers = [400]
    NN.initialize(layers)
    NN.train()
   
    if input("Wanna save model? Y/n ") !='n' :
        NN.save_model()

        
elif input("Wanna load from file? Y/n ")!='n' :
    NN.load_model()

else :
    print("Exit")
    exit()  
    
    
# Fit the model
# Final evaluation of the model
if input("Wanna test model? y/N ") =='y':
    NN.test()

#space for insertion of numpy array from user:
while input("Wanna input some stuff? Y/n ")!='n':

    NN.predict(input_stuff())
    
