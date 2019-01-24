from keras.datasets import cifar100
import random
from keras.models import Model
import numpy as np
import keras.backend as K


def load_data( cifar_model, cifar_layer, img_emb_model, num_classes = 100):
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
    np.random.shuffle(final_inds)
    Xs = x_train[final_inds]
    Ys = y_train[final_inds]
    flatten_layer = Model(inputs = cifar_model.input, outputs = cifar_model.get_layer('flatten_1').output)
    other_layer = Model(inputs = cifar_model.input, outputs = cifar_model.get_layer(cifar_layer).output)
    X_flatten = flatten_layer.predict(Xs)
    X_cnn_layer = other_layer.predict(Xs)
    X_emb = img_emb_model.predict(X_flatten)
    return X_cnn_layer, X_emb, Ys
    
    
def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    param y_true: TensorFlow tensor
    param y_pred: TensorFlow tensor of the same shape as y_true
    return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    
def lr_schedule(epoch):
    lrate = 0.005
    if epoch > 10000:
        lrate = .00005
    elif epoch > 5000 :
        lrate = 0.0001
    elif epoch > 2000:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.001       
    return lrate
