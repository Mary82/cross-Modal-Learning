from keras.datasets import cifar100
import random
from keras.models import Model
import numpy as np
import keras.backend as K


def load_cifar_flatten( cifar_model,num_classes = 100):
    random.seed(1)
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    y_pred = cifar_model.predict_classes(x_train)
    wrong_pred = [i for i,v in enumerate(y_pred) if y_pred[i]!=y_train[i]]
    final_inds = []
    subjects = np.unique(y_train).tolist()
    for subj in subjects:
        mask = y_train == subj
        inds = np.where(mask)[0]
        inds = list(set(inds)- set(wrong_pred))
        final_inds.extend(random.sample(inds, 100))
    Xs = x_train[final_inds]
    Ys = y_train[final_inds]
    pre_model = Model(inputs = cifar_model.input, outputs = cifar_model.get_layer('flatten_1').output)
    X_Embed = pre_model.predict(Xs)
    return X_Embed, Ys
    
def triplet_loss( embd, alpha = .2):
    
    a, p, n = embd[0], embd[1], embd[2]
    
    pos_dist = K.sum(K.square(a - p),axis = -1)
    neg_dist = K.sum(K.square(a - n), axis = -1)
    basic_loss = pos_dist- neg_dist+alpha
    loss = K.sum(K.maximum(0.0,basic_loss))
    
    return loss

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def accuracy(y_true, y_pred):
    return K.mean(y_pred <= 0)



