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
import pygame, random


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

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))    

#space for insertion of numpy array from user:
if input("Wanna input some stuff? y/n ")=='y':

    width = 300
    height = 300
    screen = pygame.display.set_mode((width,height))
    array=numpy.full((28,28),0)
    draw_on = False
    last_pos = (0, 0)
    color = (255, 255, 255)
    radius = 0

    def roundline(srf, color, start, end, radius=1):
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            array[int((y*28)/width)][int((x*28)/height)]=1
            pygame.draw.circle(srf, color, (x, y), radius)

            

    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == pygame.MOUSEBUTTONDOWN:
                # color = (255, 255, 255)
                # pygame.draw.circle(screen, color, e.pos, radius)
                draw_on = True
            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, e.pos, radius)
                    roundline(screen, color, e.pos, last_pos,  radius)
                last_pos = e.pos
            pygame.display.flip()

    except StopIteration:
        pass

    pygame.quit()
    print(array)
    
    x_to_predict = numpy.reshape(array, (1,784))
    predicted = model.predict(x_to_predict,batch_size=1,verbose=1)
    print("Predicted: ", numpy.argmax(predicted))



