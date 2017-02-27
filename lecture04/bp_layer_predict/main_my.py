import bp_layer.processing_data as proce
import network as net_
import numpy as np
import matplotlib.pylab as plt
def norm(m_data):
    [m_x,m_y]=m_data.shape
    max_a=np.zeros([1,m_y])
    min_a=np.zeros([1,m_y])
    for i in range(m_y):
        max_a[0,i]=np.max(m_data[:,i])
        min_a[0,i]=np.min(m_data[:,i])
        m_=max_a[0,i]-min_a[0,i]
        m_data[:,i]=(m_data[:,i]-min_a[0,i])*1.0/m_
    return m_data




if __name__=="__main__":
    data_2d=np.loadtxt("RegData2D.txt",dtype=str)
    [x_,y_]=data_2d.shape
    data_=np.zeros([x_-1,y_])
    for i in range(1,x_):
        for j in range(y_):
            data_[i-1,j]=float(data_2d[i,j])
    data_=norm(data_)
    x_a=np.array([data_[:,0]]).transpose()
    y_a=np.array([data_[:,1]]).transpose()
    x_train=x_a[0:700,:]
    y_train=y_a[0:700,:]
    x_test=x_a[0:1000,:]
    y_test=y_a[0:1000,:]
    net_1=net_.network([1,10,1],0.3, 0.3,2061, x_train,y_train ,x_test ,y_test )
    [mse_test, r_test, mse_train, r_train]=net_1.net()
    print mse_test
    print r_test
    print mse_train
    print r_train
    xx=range(1,2062)
    plt.figure(figsize=(11,8))
    plt.plot(xx,mse_test)
    plt.plot(xx,mse_train)
    plt.legend(['test_mse','train_mse'])
    plt.figure(figsize=(11,8))
    plt.plot(xx,r_test)
    plt.plot(xx,r_train)
    plt.legend(['R_test','R_train'])
    plt.show()