from __future__ import division
import numpy as np
import copy
from scipy.optimize import minimize
import matplotlib.pylab as plt
class svm_smo:
    def __init__(self, m_data_x, m_data_y, C):
        self.m_data_x=m_data_x
        self.m_data_y=m_data_y
        self.C=C
    @staticmethod
    def RBF(da_1,da_2):
        yt=(da_1-da_2)
        yy=np.dot(yt, yt.T)
        return np.exp(-0.5*yy)


    def smo(self):
        [ax,ay]=self.m_data_x.shape
        a_a=np.zeros([1,ax])
        b=0
        j=0
        kk=100
        while(kk):
            while(j<ax):
                fx=0
                for i in range(ax):
                    fx=fx+a_a[0,i]*self.m_data_y[i,0]*svm_smo.RBF(np.array([self.m_data_x[j,:]]),np.array([self.m_data_x[i,:]]))
                fx2=fx+b
                a1=0
                a2=0

                fx3=fx2*self.m_data_y[j,0]
                flag=0
                if fx3<=1 and a_a[0,j]<self.C:
                    a1=a_a[0,j]
                    flag=1
                if fx3>=1 and a_a[0,j]>0:
                    a1=a_a[0,j]
                    flag=1
                if fx3==1 and (a_a[0,j]==0 or a_a[0,j]==self.C):
                    a1=a_a[0,j]
                    flag=1
                if kk>=2 and 0<a_a[0,j]<self.C:
                    a1=a_a[0,j]
                    flag=1
                if flag==0:
                    j=j+1
                else:
                    j=j+2
                    if j+1>=ax:
                        break
                    else:
                        a2=a_a[0,j+1]
                        fx3=0
                        for i in range(ax):
                            fx3=fx3+a_a[0,i]*self.m_data_y[i,0]*svm_smo.RBF(np.array([self.m_data_x[j+1,:]]),np.array([self.m_data_x[i,:]]))
                        f5=fx3+b
                        e12=self.m_data_y[j+1,0]*((fx2-self.m_data_y[j,0])-(f5-self.m_data_y[j+1,0]))\
                            *1.0/(svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j,:])+svm_smo.RBF(self.m_data_x[j+1,:],self.m_data_x[j+1,:])
                            -2*svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j+1,:]))

                        if self.m_data_y[j,0]==self.m_data_y[j+1,0]:
                            L=max(0, a2+a1-self.C)
                            H=min(self.C, a2+a1)
                        else:
                            L=max(0, a2-a1)
                            H=min(self.C, self.C+a2-a1)
                        a3=copy.deepcopy(a2)
                        a2=a2+e12
                        if a2>H:
                            a_a[0,j+1]=H
                        if L<=a2<=H:
                            a_a[0,j+1]=a2
                        if a2<L:
                            a_a[0,j+1]=L
                        a_a[0,j]=a1+self.m_data_y[j,0]*self.m_data_y[j+1,0]*(a3-a_a[0,j+1])
                        b2=copy.deepcopy(b)
                        fx_2=0
                        fx_7=0
                        for i in range(ax):
                            fx_2=fx_2+a_a[0,i]*self.m_data_y[i,0]*svm_smo.RBF(np.array([self.m_data_x[j,:]]),np.array([self.m_data_x[i,:]]))
                            fx_7=fx_7+a_a[0,i]*self.m_data_y[i,0]*svm_smo.RBF(np.array([self.m_data_x[j+1,:]]),np.array([self.m_data_x[i,:]]))
                        fx_7=fx_2+b
                        fx_8=fx_7+b
                        if 0<a_a[0,j+1]<self.C:
                            b=b2-(fx_8-self.m_data_y[j+1,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j,:])-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j+1,:])
                        if 0<a_a[0,j]<self.C:
                            b=b2-(fx_7-self.m_data_y[j,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j,:])-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*svm_smo.RBF(self.m_data_x[j+1,:],self.m_data_x[j+1,:])
                        else:
                            b=(b2-(fx_8-self.m_data_y[j+1,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j,:])-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j+1,:])+b2-(fx_7-self.m_data_y[j,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *svm_smo.RBF(self.m_data_x[j,:],self.m_data_x[j,:])-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*svm_smo.RBF(self.m_data_x[j+1,:],self.m_data_x[j+1,:]))*1.0/2
            kk=kk-1
        return  a_a,b

def gzh(data):
    [x_,y_]=data.shape
    for j in range(y_):
        min_=np.min(data[:,j])
        max_=np.max(data[:,j])
        ad=max_-min_
        data[:,j]=1.0*(data[:,j]-min_)/ad
    return data
def classifer(a_a, b, data_x, data_y):
    [xz, xy]=data_x.shape
    x2_1=np.zeros([xz,1])
    for j in range(xz):
        fx=0
        for i in range(xz):
            fx=fx+a_a[0,i]*data_y[i,0]*svm_smo.RBF(np.array([data_x[j,:]]),np.array([data_x[i,:]]))
        x2_1[j,0]=fx+b

    pos=(x2_1>0)
    neg=(x2_1<0)
    x2_1[pos]=1
    x2_1[neg]=-1

    len_=len(data_y)
    cout=0
    for i in range(len_):
        if np.sum(abs(x2_1[i,:]-data_y[i,:]))==0:
            cout=cout+1
    return cout*1.0/len_


if __name__=="__main__":
    C=15
    test=np.loadtxt("height_and_weight.txt")
    #test[:,:-1]=gzh(test[:,:-1])
    m_data_x=test[:,:-1]
    m_data_y=test[:,-1]
    pos=(m_data_y==1)
    neg=(m_data_y==-1)

    #print m_data_y
    m_data_y=np.array([m_data_y]).T

    sv1=svm_smo(m_data_x, m_data_y, C)
    a_a,b=sv1.smo()
    [ax, ay]=m_data_x.shape
    fx=np.zeros([1,ay])
    prec=classifer(a_a,b, m_data_x, m_data_y)
    print prec
    for i in range(ax):
        fx[0,:]=fx[0,:]+a_a[0,i]*m_data_y[i,0]*np.array([m_data_x[i,:]])
    min_0=np.min(m_data_x[:,0])
    max_0=np.max(m_data_x[:,0])
    min_1=np.min(m_data_x[:,1])
    max_1=np.max(m_data_x[:,1])
    ac1=np.linspace(min_0, max_0, 1000)
    ac2=np.linspace(min_1, max_1, 1000)
    dit=np.zeros([1000000,2])
    ak=0
    for ai in ac1:
        for aj in ac2:
            dit[ak,0]=ai
            dit[ak,1]=aj
            ak=ak+1

    x2_1=np.zeros([1000000,2])
    hj=0
    for j in range(1000000):
        fx=0
        for i in range(ax):
            fx=fx+a_a[0,i]*m_data_y[i,0]*svm_smo.RBF(np.array([dit[j,:]]),np.array([m_data_x[i,:]]))
        if abs(fx+b)<0.01:
            x2_1[hj,:]=dit[j,:]
            hj=hj+1
    print hj

    plt.plot(m_data_x[pos][:,0],m_data_x[pos][:,1],'b*')

    plt.plot(m_data_x[neg][:,0],m_data_x[neg][:,1],'ro')
    plt.legend(['gril','boy'], loc='best')
    print x2_1[0:hj,:]
    plt.plot(x2_1[0:hj,0],x2_1[0:hj,1],'k.')
    plt.show()