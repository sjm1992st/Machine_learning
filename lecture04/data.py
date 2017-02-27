from __future__ import division
import numpy as np
import matplotlib.pylab as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from scipy.optimize import leastsq
def mse(y_r,y):
    m_se=np.mean((y-y_r)**2)
    return m_se
def R2(y_r,y):
    r=1-1.0*np.sum((y-y_r)**2)/np.sum((y_r-np.mean(y_r))**2)
    return r

def func(p, x):
    f = np.poly1d(p)
    return f(x)

def residuals(p, y, x):
    ret=y - func(p, x)
    ret = np.append(ret, 0.5*p**2)
    return ret
if __name__=="__main__":
    data_2d=np.loadtxt("RegData2D.txt",dtype=str)
    [x_,y_]=data_2d.shape
    data_=np.zeros([x_-1,y_])
    for i in range(1,x_):
        for j in range(y_):
            data_[i-1,j]=float(data_2d[i,j])
    x_a=data_[:,0]
    y_a=data_[:,1]
    print data_2d.shape
    plt.scatter(x_a, y_a, s=5)
    #plt.show()
    for m in [2,3,11,15,16,18]:
        p0 = np.random.randn(m)
        plsq = leastsq(residuals, p0, args=(y_a[0:700], x_a[0:700]))
        x_show = np.linspace(-1, 5, 1000)
        #print 'Fitting Parameters :', plsq[0]
        y=func(plsq[0], x_a)
        #print y
        m_se=mse(y_a,y)
        r=R2(y_a,y)
        print m_se,r
        plt.plot(x_show, func(plsq[0], x_show),label='fitted curve')
    plt.legend(['1', '2', '10','14','15','17'], loc='upper left')
    plt.show()
