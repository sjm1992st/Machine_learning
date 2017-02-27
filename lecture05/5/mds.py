from __future__ import division
import numpy as np
import copy
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
def dist_(m_data):
    [m_x,m_y]=m_data.shape
    dist=np.zeros([m_x,m_x])
    for i in range(m_x):
        for j in range(m_x):
            if i!=j:
                dd=m_data[i,:]-m_data[j,:]
                dist[i,j]=np.dot(dd,dd.T)**0.5
    return dist



def find_v(new_a,mm):
    [xc,yc]=new_a.shape
    sd=0
    for i in range(yc):
        if mm==new_a[0,i]:
            sd=i
            break
    return sd


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

def creat_b(m_data):
    [m_x,m_y]=m_data.shape
    dist=dist_(m_data)
    b=np.zeros([m_x,m_x])
    d=np.sum(dist**2)*1.0/(m_x**2)
    for i in range(m_x):
        for j in range(m_x):
            b[i,j]=-0.5*(dist[i,j]**2-np.sum(dist[i,:]**2)*1.0/m_x-np.sum(dist[:,j]**2)*1.0/m_x+d)
    return b


if __name__=="__main__":
    num=3
    m_data=np.loadtxt("CAT4D3GROUPS.txt")
    alen=len(m_data)
    b=creat_b(m_data)
    v,r=np.linalg.eig(b)
    v=np.array([v])
    v=v.real
    [v_new, lis]=sort_v(v,num)
    r_new=r_new(r, lis)

    mds=np.dot(r_new, v_new**0.5)
    mds_=mds.real

    aq1=m_data[0,:]-m_data[1,:]
    aq2=mds_[0,:]-mds_[1,:]
    print mds_[0,:]
    print np.sum(aq1**2)
    print np.sum(aq2**2)
    np.savetxt("mds.txt",mds_)
    ax=plt.subplot(111,projection='3d')
    ax.scatter(mds_[0:alen-1,0],mds_[0:alen-1,1],mds_[0:alen-1,2],c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()