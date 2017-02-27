import sys,re,os
import numpy as np
from scipy import stats
import matplotlib.pylab as plt
import import_hw as dat

def draw_kde(mdata,m_data_y):
    #print m_data_y.shape
    [num, dim]=mdata.shape
    pos=(m_data_y[:,-1]==1)
    neg=(m_data_y[:,-1]==-1)
    if dim==1:
        for i in range(2):
            if i==0:
                fs=pos
            if i==1:
                fs=neg
            xmin=np.min(mdata[fs][:,0])
            xmax=np.max(mdata[fs][:,0])
            #print mdata.shape
            gkde = stats.kde.gaussian_kde(mdata[fs].T)
            ind = np.arange(xmin,xmax,0.1)
            if i==0:
                plt.plot(ind, gkde(ind), label='gril weight', color="g")
            if i==1:
                plt.plot(ind, gkde(ind), label='boy weight', color="r")
            plt.title('Kernel Density Estimation')
            plt.legend()
    if dim==2:
        for i in range(2):
            plt.figure(figsize=(8,7))
            if i==0:
                fs=pos
            if i==1:
                fs=neg
            kde = stats.kde.gaussian_kde(mdata.T)
            x_flat = np.r_[mdata[fs][:,0].min():mdata[fs][:,0].max():128j]
            y_flat = np.r_[mdata[fs][:,1].min():mdata[fs][:,1].max():128j]
            #X, Y = np.mgrid[mdata[:,0].min():mdata[:,0].max():128j, mdata[:,1].min():mdata[:,1].max():128j]

            x,y = np.meshgrid(x_flat,y_flat)
            #print x.shape
            grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
            z = kde(grid_coords.T)
            z = z.reshape(128,128)
            #print cmap=plt.cm.gist_earth_r,
            if i==0:
                plt.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),extent=[mdata[fs][:,0].min(),mdata[fs][:,0].max(), mdata[fs][:,1].min(),mdata[fs][:,1].max()])
                plt.title('gril_2dKernel Density Estimation')
            if i==1:
                plt.imshow(z,aspect=x_flat.ptp()/y_flat.ptp(),extent=[mdata[fs][:,0].min(),mdata[fs][:,0].max(), mdata[fs][:,1].min(),mdata[fs][:,1].max()])
                plt.title('boy_2dKernel Density Estimation')
        plt.show()

if __name__ == '__main__':
    m_data_x,m_data_y=dat.import_data()

    data_=np.array([m_data_x[:,1]]).T
    draw_kde(data_,m_data_y)

    draw_kde(m_data_x,m_data_y)