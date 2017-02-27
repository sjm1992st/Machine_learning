from __future__ import division
import numpy as np
import load_and_extract_mnist_data as im
def my_data(mdata):
    [m_x, m_y, m_z, m_j]=mdata.shape
    new_data=np.zeros([m_x,m_z*m_j])
    for i in range(m_x):
        new_data[i,:]=mdata[i][0].reshape((1, m_z*m_j))
    #ad=np.ones([m_x, 1])
    #new_data=np.c_[new_data,ad]
    return  new_data
def y_lab(mm_y):
    l_a=len(mm_y)
    y_re=np.zeros([l_a,10])
    for i in range(l_a):
        y_re[i,mm_y[i]]=1
    return y_re

def main_data():
    X_train, y_train, X_val, y_val, X_test, y_test = im.load_dataset()
    m_Xtrain=my_data(X_train)  ##### (50000,785)
    m_Xval=my_data(X_val)      ##### (10000, 785)
    m_Xtest=my_data(X_test)   #### (10000, 785)
    mm_xtrain=m_Xtrain[0:50000,:]
    mm_ytrain=y_train[0:50000]
    mm_xval=m_Xval[0:10000,:]
    mm_yval=y_val[0:10000]
    train_y=y_lab(mm_ytrain)
    val_y=y_lab(mm_yval)
    return mm_xtrain, train_y, mm_xval, val_y