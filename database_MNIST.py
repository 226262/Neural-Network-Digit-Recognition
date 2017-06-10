from keras.datasets import mnist
from keras.utils import np_utils

class Database_MNIST :
    
    def __init__(self):
        print("database_init")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.num_pixels = self.X_train.shape[1] * self.X_train.shape[2]
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.num_pixels).astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.num_pixels).astype('float32')
        
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]
    
        self.X_test= self.X_test / 255
        self.X_train= self.X_train / 255
