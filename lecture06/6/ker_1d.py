from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import import_hw as dat


def x_i(l_dim, h_dim, start_dim):
    [z_x, z_y]=l_dim.shape
    list_x=[]
    sl=1
    for i in range(z_y):
        xx=[]
        n=int(l_dim[0,i]/h_dim[0,i])
        sd=l_dim[0,i]%h_dim[0,i]
        if sd>0:
            n=n+1
        if sd==0:
            sd=h_dim[0,i]
        x_i=np.zeros([1,n])
        for j in range(n-1):
            x_i[0,j]=start_dim[0,i]+0.5*h_dim[0,i]+j*h_dim[0,i]
            xx.append(x_i[0,j])
        x_i[0,n-1]=x_i[0,j]+0.5*h_dim[0,i]+sd*1.0/2
        print x_i
        xx.append(x_i[0,n-1])
        sl=sl*n
        list_x.append(xx)

    m_x=np.zeros([sl, z_y])
    mn=0
    if z_y==1:
        for jl in list_x[0]:
            m_x[mn, 0]=jl
            mn=mn+1
    if z_y==2:
        for j_1 in list_x[0]:
            for j_2 in list_x[1]:
                m_x[mn, 0]=j_1
                m_x[mn, 1]=j_2
                mn=mn+1
    return m_x

def RBF(mdata, h):
    [num, dim]=mdata.shape
    l_dim=np.zeros([1,dim])
    h_dim=np.zeros([1,dim])
    start_dim=np.zeros([1,dim])
    s_h=1
    for j in range(dim):
        min_0=np.min(mdata[:,j])
        max_0=np.max(mdata[:,j])
        start_dim[0,j]=min_0
        fw=max_0-min_0
        l_dim[0,j]=fw
        h_dim[0,j]=h*np.std(mdata[:,j])
        #n=int(fw/h_dim[0,j])
        #sd=fw%h
        #if sd>0:
            #n=n+1
        s_h=s_h*h_dim[0,j]
    mx=x_i(l_dim, h_dim, start_dim)
    [cx,cy]=mx.shape
    ker=np.zeros([cx,1])
    for an in range(cx):
        asd=0
        for i in range(num):
            #ss=1
            #for gh in range(cy):
                #print h_dim[0,gh]
            ss=np.sqrt(2*np.pi)*np.exp((-0.5*(mx[an,0]-mdata[i,0])/h_dim[0,0])**2)
            asd=asd+ss
        ker[an,0]=asd*1.0/s_h/num
    return mx,ker



if __name__=="__main__":
    m_data_x,m_data_y=dat.import_data()
    data_=np.array([m_data_x[:,0]]).T
    h=0.05
    mx,kerr=RBF(data_, h)
    plt.plot(mx,kerr)
    plt.show()