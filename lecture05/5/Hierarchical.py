from __future__ import division
import numpy as np
import random
import copy
import matplotlib.pylab as plt

def list_array(wd, clo):
    len_=len(wd)
    xc=np.zeros([len_, clo])
    for i in range(len_):
        ad=wd[i]
        xc[i, :]=ad
    return xc

def c_dist(w_a, w_b,clo):
    m_a=len(w_a)
    m_b=len(w_b)
    s_a=list_array(w_a, clo)
    s_b=list_array(w_b, clo)
    sum_dist=0
    for i in range(m_a):
        for j in range(m_b):
            dd=s_a[i,:]-s_b[j,:]
            ad=np.dot(dd,dd.T)
            sum_dist=sum_dist+ad
    sum_dist=sum_dist*1.0/m_a/m_b
    return sum_dist

def hierarchical(cat2d,k):
    [p,clo]=cat2d.shape
    w_d = [[] for n in range(p)]
    for i in range(p):
        w_d[i].append(cat2d[i,:])
    while(p>k):
        dist=np.zeros([p,p])
        for i in range(p):
            for j in range(p):
                if i<j:
                    dist[i,j]=c_dist(w_d[i],w_d[j],clo)
                elif i==j:
                    dist[i,j]=float("inf")
                else:
                    dist[i,j]=dist[j,i]
        #print np.min(dist)
        [ax,ay]=np.where(dist==np.min(dist))
        w_d[ax[0]]=w_d[ax[0]]+w_d[ay[0]]
        w_d=w_d[0:ay[0]]+w_d[ay[0]+1:]
        p=p-1
    return w_d
def plot_k(wd, k, clo):
    for i in range(k):
        wdd=list_array(wd[i], clo)
        plt.plot(wdd[:,0],wdd[:,1],'*')

if __name__=="__main__":
    cat2d=np.loadtxt("pca.txt")
    [len_, clo]=cat2d.shape
    k=5
    wd=hierarchical(cat2d,k)
    plot_k(wd, k, clo)
    plt.legend(['1','2', '3','4','5'])
    plt.show()
