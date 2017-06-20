from keras.datasets import mnist
from keras.utils import np_utils

def make_discrete(number_to):

    if(number_to < 0.2):
        return 0
    else:
        return 1

class Database_MNIST :
    

    def __init__(self):
        print("database_init")

        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.num_pixels = self.X_train.shape[1] * self.X_train.shape[2]
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.num_pixels).astype('bool')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.num_pixels).astype('bool')
        
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]



        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.num_pixels).astype('int')
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.num_pixels).astype('int')

        # for i in range(59999) :
        #     for x in range(783):
        #         self.X_train[i][x]= make_discrete(self.X_train[i][x] / 255)

        # for j in range(9999) :
        #     for y in range(783):
        #         self.X_test[j][y]= make_discrete(self.X_test[j][y] / 255)


        # self.X_test= make_discrete( self.X_test / 255)
     
        # self.X_test = self.X_test /255
        # self.X_train = self.X_train /255

        # print(self.X_test[0])
