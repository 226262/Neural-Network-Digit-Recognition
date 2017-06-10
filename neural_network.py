import os
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import matplotlib.pyplot as plt
from database_MNIST import Database_MNIST


class Neural_Network :

    model = Sequential()

    mnist = Database_MNIST()

    def initialize(self,hidden_layers):

        self.model.add(Dense(self.mnist.num_pixels, input_dim=self.mnist.num_pixels, kernel_initializer='normal', activation='relu')) #hidden layer

        for number in hidden_layers :
            print("adding layer: ",number)
            self.model.add(Dense(number,  activation='relu'))

        self.model.add(Dense(self.mnist.num_classes, kernel_initializer='normal', activation='sigmoid'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        
    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")         

    def load_model(self):

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
        self.model=loaded_model
        print("Loaded model from disk")
        
    def train (self):
        self.model.fit(self.mnist.X_train, self.mnist.y_train, validation_data=(self.mnist.X_test, self.mnist.y_test), epochs=3, batch_size=200, verbose=1)
        
    def test(self):
        scores = self.model.evaluate(self.mnist.X_test, self.mnist.y_test, verbose=1)
        print("Baseline Error: %.2f%%" % (100-scores[1]*100))  

    
    def predict(self,arrayek):
        x_to_predict = numpy.reshape(arrayek, (1,784))
        predicted = self.model.predict(x_to_predict,batch_size=1,verbose=1)
        print("Predicted: ", numpy.argmax(predicted))
        
