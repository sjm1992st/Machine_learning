import numpy as np
def import_data():
    test=np.loadtxt("height_and_weight.txt")
    #test[:,:-1]=gzh(test[:,:-1])
    m_data_x=test[:,:-1]
    m_data_y=test[:,-1]
    pos=(m_data_y==1)
    neg=(m_data_y==-1)
    m_data_y=np.array([m_data_y]).T
    return m_data_x,m_data_y