from keras.datasets import cifar100
from keras.utils import np_utils
import random
from keras.models import Model
import numpy as np
import keras.backend as K



def load_data(num_classes):
	(x_train, y_train), (x_test, y_test) = cifar100.load_data()
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
 	x_train /= 255.
	x_test /= 255.
	y_train = np_utils.to_categorical(y_train,num_classes)
	y_test = np_utils.to_categorical(y_test,num_classes)
	
	return (x_train, y_train), (x_test, y_test)

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 5000:
        lrate = 0.00005
    elif epoch > 2500:
        lrate = 0.0001
    elif epoch > 1000:
    	lrate = 0.0005       
    return lrate


