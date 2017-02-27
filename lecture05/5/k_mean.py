from __future__ import division
import numpy as np
import random
import copy
import matplotlib.pylab as plt
def kmean(cat2d, k):
    [len_, clo]=cat2d.shape
    num=range(len_)
    random.shuffle(num)
    num=num[0:k]
    mean_=np.zeros([k,clo])
    mk=0
    for aa in num:
        mean_[mk, :]=cat2d[aa,:]
        mk=mk+1
    while(1):
        w_d = [[] for n in range(k)]
        for i in range(len_):
            d_l=[]
            for j in range(k):
                dd=cat2d[i,:]-mean_[j,:]
                ad=np.dot(dd,dd.T)
                d_l.append(ad)
            wz=d_l.index(min(d_l))
            w_d[wz].append(cat2d[i,:])
        flag=0
        for aj in range(k):
            if sum(abs(mean_[aj, :]-np.mean(w_d[aj],0)))!=0:
                mean_[aj, :]=np.mean(w_d[aj],0)
                flag=1
        if flag==0:
            break
    return w_d,mean_

def list_array(wd, clo):
    len_=len(wd)
    xc=np.zeros([len_, clo])
    for i in range(len_):
        ad=wd[i]
        xc[i, :]=ad
    return xc

def plot_k(wd, k, clo):
    for i in range(k):
        wdd=list_array(wd[i], clo)
        plt.plot(wdd[:,0],wdd[:,1],'*')
def dist(wd, mean_a, clo):
    sum=0
    for j in range(k):
        wdd=list_array(wd[j], clo)
        len_=len(wd[j])
        for i in range(len_):
            dd=wdd[i,:]-mean_a[j,:]
            ad=np.dot(dd, dd.T)
            sum=sum+ad
    return sum


if __name__=="__main__":
    cat2d=np.loadtxt("pca.txt")
    [len_, clo]=cat2d.shape
    unif=np.random.rand(len_, clo)
    ld=[]
    ld_u=[]
    gk=range(1,9)
    for k in [5]:
        [wd, mean_]=kmean(cat2d, k)
        [wd_u, mean_u]=kmean(unif, k)
        da=dist(wd, mean_, clo)
        da_u=dist(wd_u, mean_u, clo)
        ld.append(da)
        ld_u.append(da_u)
    #plt.plot(gk, ld, marker='*')
    #plt.plot(gk, ld_u, marker='.')
    #plt.legend(['cat2d','uniform'])
    print da
    k=5
    plot_k(wd, k, clo)
    plt.legend(['1','2', '3','4', '5'])
    plt.show()

