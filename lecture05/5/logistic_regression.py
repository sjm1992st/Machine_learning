from __future__ import division
import numpy as np
import copy
from scipy.optimize import minimize
import matplotlib.pylab as plt
def exp_m(dt, wt):
    #print wt.shape
    check=-np.dot(dt, wt)
    #print check
    #pp=(check>300)
    #check[pp]=300
    #print check.shape
    zd=1.0/(1 + np.exp( check ))
    #print 1.0/(1 + np.exp( -np.dot(dt, wt) ))
    #bo_1=(zd==1)
    #bo_2=(zd==0)
    #zd[bo_1]=0.99
    #zd[bo_2]=0.01
    return zd
def logistic(m_data_x, m_data_y,step, choose):
    [m_x,m_y]=m_data_x.shape
    zone=np.ones([m_x,1])
    m_data_x=np.c_[m_data_x, zone]
    wt=np.ones([m_y+1,1])

    while(step):
        zz=0
        ax=0
        #for i in range(m_x):
            #lx=np.dot(m_data_y.T, np.log(exp_m(m_data_x, wt)))+ np.dot((1-m_data_y).T, np.log(1-exp_m(m_data_x, wt)))
            #ax=ax+lx[0,0]
        #print ax
        #print np.log(1-exp_m(m_data_x, wt))
        #print exp_m(m_data_x, wt).shape
        zaz=m_data_y-exp_m(m_data_x, wt)
        zz=np.dot(m_data_x.T,(m_data_y-exp_m(m_data_x, wt)))*1.0/m_x
        #zz=np.array([zz]
        if choose==2:
            zz=zz-0.005*wt
        wt=wt+0.001*zz
        #print wt.shape
        step=step-1
    return wt

def gzh(data):
    [x_,y_]=data.shape
    for j in range(y_):
        min_=np.min(data[:,j])
        max_=np.max(data[:,j])
        ad=max_-min_
        data[:,j]=1.0*(data[:,j]-min_)/ad
    return data
def classifer(ww, data_x, data_y):
    [m_x,m_y]=data_x.shape
    zone=np.ones([m_x,1])
    data_x=np.c_[data_x, zone]
    my=exp_m(data_x, ww)
    #print my
    pos=(my>0.5)
    neg=(my<0.5)
    my[pos]=1
    my[neg]=0

    len_=len(data_y)
    cout=0
    for i in range(len_):
        if np.sum(abs(my[i,:]-data_y[i,:]))==0:
            cout=cout+1
    return cout*1.0/len_
def my_plot(m_data_x, m_data_y,step, choose, dim):
    plt.figure(figsize=(8,6))

    min_0=np.min(m_data_x[:,0])
    max_0=np.max(m_data_x[:,0])
    [sx,sy]=m_data_x.shape
    line=np.zeros([sx,1])
    ac=np.linspace(min_0, max_0, 100)
    ac=np.array([ac])
    x2_1=np.zeros([1,100])


    if dim==2:
        ww2=logistic(m_data_x, m_data_y,step, choose)
        pec2=classifer(ww2, m_data_x, m_data_y)
        print pec2
        print ww2
        for i in range(100):
            #print fx[0,0]
            x2_1[0,i]=(-ww2[2,0]-ac[0,i]*ww2[0,0])*1.0/ww2[1,0]
        plt.plot(m_data_x[pos][:,0],m_data_x[pos][:,1],'b*')
        plt.plot(m_data_x[neg][:,0],m_data_x[neg][:,1],'ro')
        plt.plot(ac[0,:],x2_1[0,:],linewidth=2)
    if dim==1:
        ww1=logistic(m_data_x[:,1:2], m_data_y,step, choose)
        pec1=classifer(ww1, m_data_x[:,1:2], m_data_y)
        print pec1
        x2_2=-ww1[1,0]*1.0/ww1[0,0]
        plt.plot(m_data_x[pos][:,0],'b*')
        plt.plot(m_data_x[neg][:,0],'ro')
        print x2_2
        plt.plot(x2_2,'y+')
    plt.legend(['gril','boy'], loc='best')
    plt.show()
if __name__=="__main__":
    test=np.loadtxt("hwlogistic.txt")
    #test[:,:-1]=gzh(test[:,:-1])
    m_data_x=test[:,:-1]
    m_data_y=test[:,-1]
    pos=(m_data_y==1)
    neg=(m_data_y==0)
    m_data_y=np.array([m_data_y]).T
    step=1500


    my_plot(m_data_x, m_data_y,step, 1, 1) ##ml
    my_plot(m_data_x, m_data_y,step, 2, 2) ##map a=0.1






    """
    ww=logistic(m_data_x, m_data_y,step, choose)
    pec=classifer(ww, m_data_x, m_data_y)
    print pec
    min_0=np.min(m_data_x[:,0])
    max_0=np.max(m_data_x[:,0])

    ac=np.linspace(min_0, max_0, 100)
    ac=np.array([ac])
    x2_1=np.zeros([1,100])
    for i in range(100):
        #print fx[0,0]
        x2_1[0,i]=(-ww[2,0]-ac[0,i]*ww[0,0])*1.0/ww[1,0]
    plt.plot(m_data_x[pos][:,0],m_data_x[pos][:,1],'b*')
    plt.plot(m_data_x[neg][:,0],m_data_x[neg][:,1],'ro')
    plt.legend(['gril','boy'], loc='best')
    plt.plot(ac[0,:],x2_1[0,:],linewidth=2)
    """