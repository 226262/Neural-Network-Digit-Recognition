#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import model_from_json
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#plt.imshow(X_test[4], cmap=plt.get_cmap('gray'))
#plt.show()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
    # create model
    model = Sequential() 
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu')) #hidden
    model.add(Dense(400,  activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model

# build the model

model = baseline_model()

if input("Wanna train model? y/n ")=='y':
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=1)
    
    if input("Wanna save model? y/n ") =='y' :
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")        

elif input("Wanna load from file? y/n ")=='y' :
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    model=loaded_model
    print("Loaded model from disk")
else :
    print("Exit")
    exit()  
    
    
# Fit the model
# Final evaluation of the model
if input("Wanna test model? y/n ") =='y':
    
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))


    
#space for insertion of numpy array from user:
if input("Wanna input some stuff? y/n ")=='y':

    

    
    x_to_predict = numpy.reshape(X_test[4], (1,784))
    predicted = model.predict(x_to_predict,batch_size=1,verbose=1)
    print("Predicted: ", numpy.argmax(predicted))



