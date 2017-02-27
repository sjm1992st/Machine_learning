from __future__ import division
import numpy as np
import copy
import matplotlib.pylab as plt

def sort_v(v,num):
    [ix,iy]=v.shape
    lis=[]
    v_new=np.zeros([num,num])
    v_copy=copy.deepcopy(v)
    s_v=np.sort(v_copy[0,:])
    s_v=np.array([s_v])
    k=0
    for j in range(iy-1,iy-num-1,-1):
        v_new[k,k]=s_v[0,j]
        w=find_v(v, s_v[0,j])
        lis.append(w)
        k=k+1
    return v_new, lis

def r_new(v, lis):
    len_=len(lis)
    [zx,zy]=v.shape
    v_=np.zeros([zx,1])
    for i in range(len_):
        a=lis[i]
        v_=np.c_[v_, v[:,a]]
    r_a=copy.deepcopy(v_[:,1:len_+1])
    return r_a

def find_v(new_a,mm):
    [xc,yc]=new_a.shape
    sd=0
    for i in range(yc):
        if mm==new_a[0,i]:
            sd=i
            break
    return sd


if __name__=="__main__":
    cat3d=np.loadtxt("mds.txt")
    a_cov=np.cov(cat3d.T)
    num=2
    v,r=np.linalg.eig(a_cov)
    v=np.array([v])
    v=v.real
    [v_new, lis]=sort_v(v,num)
    r_new=r_new(r, lis)
    pca_data=np.dot(cat3d, r_new)
    alen=len(pca_data)
    np.savetxt("pca.txt",pca_data)
    plt.scatter(pca_data[0:alen-1,0],pca_data[0:alen-1,1], c='r')
    plt.show()