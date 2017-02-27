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

    def smo(self):
        [ax,ay]=self.m_data_x.shape
        a_a=np.zeros([1,ax])
        b=0
        j=0
        kk=100
        while(kk):
            while(j<ax):
                fx=np.zeros([1,ay])
                for i in range(ax):
                    fx[0,:]=fx[0,:]+a_a[0,i]*self.m_data_y[i,0]*np.array([self.m_data_x[i,:]])
                a1=0
                a2=0
                fx2=np.dot(fx, self.m_data_x[j,:].T)+b

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
                        f5=np.dot(fx, self.m_data_x[j+1,:].T)+b
                        e12=self.m_data_y[j+1,0]*((fx2-self.m_data_y[j,0])-(f5-self.m_data_y[j+1,0]))\
                            *1.0/(np.dot(self.m_data_x[j,:],self.m_data_x[j,:].T)+np.dot(self.m_data_x[j+1,:],self.m_data_x[j+1,:].T)
                            -2*np.dot(self.m_data_x[j,:],self.m_data_x[j+1,:].T))

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
                        for i in range(ax):
                            fx_2=fx_2+a_a[0,i]*self.m_data_y[i,0]*self.m_data_x[i,:]
                        fx_7=np.dot(fx_2, self.m_data_x[j,:].T)+b
                        fx_8=np.dot(fx_2, self.m_data_x[j+1,:].T)+b
                        if 0<a_a[0,j+1]<self.C:
                            b=b2-(fx_8-self.m_data_y[j+1,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *np.dot(self.m_data_x[j,:],self.m_data_x[j,:].T)-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*np.dot(self.m_data_x[j,:],self.m_data_x[j+1,:].T)
                        if 0<a_a[0,j]<self.C:
                            b=b2-(fx_7-self.m_data_y[j,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *np.dot(self.m_data_x[j,:],self.m_data_x[j,:].T)-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*np.dot(self.m_data_x[j+1,:],self.m_data_x[j+1,:].T)
                        else:
                            b=(b2-(fx_8-self.m_data_y[j+1,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *np.dot(self.m_data_x[j,:],self.m_data_x[j,:].T)-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*np.dot(self.m_data_x[j,:],self.m_data_x[j+1,:].T)+b2-(fx_7-self.m_data_y[j,0])-self.m_data_y[j,0]*(a_a[0,j]-a1)\
                            *np.dot(self.m_data_x[j,:],self.m_data_x[j,:].T)-self.m_data_y[j+1,0]*(a_a[0,j+1]-a3)*np.dot(self.m_data_x[j+1,:],self.m_data_x[j+1,:].T))*1.0/2
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
    fx=np.zeros([1,ay])
    [xz, xy]=data_x.shape
    for i in range(ax):
        fx[0,:]=fx[0,:]+a_a[0,i]*m_data_y[i,0]*np.array([m_data_x[i,:]])
    x2_1=np.dot(data_x,fx.T)+b

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
    ac=np.linspace(min_0, max_0, 100)
    ac=np.array([ac])
    x2_1=np.zeros([1,100])
    for i in range(100):
        #print fx[0,0]
        x2_1[0,i]=(-b-fx[0,0]*ac[0,i])*1.0/fx[0,1]
    plt.plot(m_data_x[pos][:,0],m_data_x[pos][:,1],'b*')

    plt.plot(m_data_x[neg][:,0],m_data_x[neg][:,1],'ro')
    plt.legend(['gril','boy'], loc='best')
    plt.plot(ac[0,:],x2_1[0,:],linewidth=2)
    plt.show()