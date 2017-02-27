import numpy as np
import cPickle as pickle
from urllib import urlretrieve
import os
import gzip

def load_dataset():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Downloading MNIST dataset...")
        urlretrieve(url, filename)
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)
    X_train, y_train = data[0]
    X_val, y_val = data[1]
    X_test, y_test = data[2]
    X_train = X_train.reshape((-1, 1, 28, 28))
    X_val = X_val.reshape((-1, 1, 28, 28))
    X_test = X_test.reshape((-1, 1, 28, 28))
    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__=="__main__":
    ### database format
    ### img: [num of img,img channel,img height,img weight]; label :[num,1]
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    first_train_img = X_train[0][0]
    first_train_label = y_train[0]
    #print y_train[0:20]
    [m_x, m_y, m_z, m_j]=X_train.shape
    new_data=np.zeros([m_x,m_z*m_j])
    for i in range(m_x):
        new_data[i,:]=X_train[i][0].reshape((1, m_z*m_j))
    #print new_data[0,:],new_data[31,312]
    #print X_val.shape
    print first_train_img  ##should be 28*28
