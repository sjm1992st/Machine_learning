from __future__ import division
import numpy as np
import random
import matplotlib.pylab as plt
import math
import load_and_extract_mnist_data as im
###########################
######## weight_initialization
def weight_init(w_a,bais):
    [w_x, w_y]=w_a.shape
    d=np.zeros([w_x,w_y])
    #d=np.random.randn(w_x, w_y)
    for i in range(w_x-1):
        for j in range(w_y):
            d[i,j]=random.uniform(-1,1)
    for f in range(w_y):
        d[w_x-1,f]=bais
    return d
#########
def input_a(x_1, w_1):
    return np.dot(x_1,w_1)

#########



##############
##############
def ker_1(v):
    [v_x,v_y]=v.shape
    s=np.zeros([v_x,v_y])
    for i in range(v_y):
        s[0,i]=1.0/(1+math.exp(-1*v[0,i]))
    return s
##############
#########
def ker_2(v):
    [v_x,v_y]=v.shape
    s=np.zeros([v_x,v_y])
    for i in range(v_y):
        if v[0,i]<=0:
            s[0,i]=0
        if v[0,i]>0:
            s[0,i]=v[0,i]
    return  s

##########
def lable(out):
    [v_x,v_y]=out.shape
    s=np.zeros([v_x,v_y])
    for i in range(v_y):
        if out[0,i]<0.5:
            s[0,i]=0
        if out[0,i]>0.5:
            s[0,i]=1
    return  s

###########
def l_err(y,y_r):
    l_e=0.5*(y_r-y)**2
    return  np.sum(l_e)
###########
def diff_out_1(o_in):
    g=(1-ker_1(o_in))*ker_1(o_in)
    return g
###########
###########
def diff_out_2(o_in):
    [oi_x,oi_y]=o_in.shape
    g=np.zeros([oi_x,oi_y])
    for i in range(oi_y):
        if o_in[0,i]<=0:
            g[0,i]=0
        if o_in[0,i]>0:
            g[0,i]=1
    return g
###########

def w_out_diff(y,y_r,o_in):
    return diff_out_1(o_in)*(y_r-y)
##############
"""
def mse(w_up12,w_up23, arra,y_train1):
    [a_x, a_y]=arra.shape
    lra=0
    for k in range(int(a_x/50)):
        input_arra=np.array([arra[k,:]])      ## (1,785)

        h_in=input_a(input_arra, w_up12)         ##(1,785).(785,n)->(1,n)
        #print h_in.shape
        h_out=ker_1(h_in)                       ## (1,n)
        #print h_out
        bb=np.array([[1]])
        h_out=np.c_[h_out, bb]                  ## (1,n+1)
        o_in=np.dot(h_out, w_up23)              ## (1,n+1).(n+1,10)->(1,10)

        o_out=ker_1(o_in)

        lab_out=lable(o_out)
        y_re=np.zeros([1,10])
        y_re[0, y_train1[k]]=1
        lra=lra+l_err(lab_out,y_re)
    lra=lra*1.0/a_x
    return  lra
"""
#################


#################
def my_test(w_up12,w_up23, xtest, ytest):
    [a_x, a_y]=xtest.shape
    m_precision=0
    lra=0
    for k in range(a_x):
        input_arra=np.array([xtest[k,:]])      ## (1,785)

        h_in=input_a(input_arra, w_up12)         ##(1,785).(785,n)->(1,n)

        h_out=ker_1(h_in)                       ## (1,n)

        bb=np.array([[1]])

        h_out=np.c_[h_out, bb]                  ## (1,n+1)

        o_in=np.dot(h_out, w_up23)              ## (1,n+1).(n+1,10)->(1,10)
        o_out=ker_1(o_in)

        lab_out=lable(o_out)
        y_re=np.zeros([1,10])
        y_re[0, ytest[k]]=1
        lra=lra+l_err(lab_out,y_re)
        sub=np.sum(abs(lab_out-y_re))
        if sub==0:
            m_precision=m_precision+1
    return  lra*1.0/a_x,m_precision*1.0/a_x
##########################################################


############################################################
def bp(array_a, y_train1, X_val1, y_val1, w_12, w_23,w_34, pb, nn, nb):
    [a_x, a_y]=array_a.shape    #(2000,4)
    pb_mse=[]
    md_precision=[]
    w_up12=w_12
    w_up23=w_23
    w_up34=w_34
    """##-stochastic gradient method-##"""
    for pi in range(pb):
        #a=mse(w_up12,w_up23,array_a)
        #print a
        #pb_mse.append(a)
        lra=0
        mm_pe=0
        for k in range(a_x):
            #n= 0.4
            input_arra=np.array([array_a[k,:]])## (1,785)

            h_in=input_a(input_arra, w_up12)         ##(1,785).(785,n)->(1,n)
            #print h_in.shape
            h_out1=ker_1(h_in)                       ## (1,n)
            #print h_out
            bb=np.array([[1]])
            h_out1=np.c_[h_out1, bb]                  ## (1,n+1)
            o_in1=np.dot(h_out1, w_up23)              ## (1,n+1).(n+1,10)->(1,10)
            #print o_ino_in
            o_out1=ker_1(o_in1)
            ################
            h_out2=np.c_[o_out1, bb]                  ## (1,n+1)
            o_in2=np.dot(h_out2, w_up34)
            o_out2=ker_1(o_in2)
            #print  o_out
            lab_out=lable(o_out2)                  ### (1,10)

            y_re=np.zeros([1,10])
            y_re[0, y_train1[k]]=1
            yr=w_out_diff(lab_out, y_re, o_in2)   ### (1,10)- deviation
            diff_o=np.dot(yr.T,h_out2)            ### (10,1).(1,n+1) --(10,n+1)
            #print diff_o.shape
            #print yr
            """
            ##gradient check
            """
            if k==0 and pi==5:
                ee=l_err(lab_out,array_a[k,3]) ###(1,10)
                w_e=np.zeros([nb+1,10])
                w_e[0,0]=10**-7
                w_e=w_up23+w_e
                o_e=np.dot(h_out2, w_e)
                o_oe=ker_1(o_e)
                lab_e=lable(o_oe)
                ee2=l_err(lab_e,array_a[k,3])
                w_check=(ee-ee2)*1.0/(10**-7)
                #print w_check-diff_o.T[0,0]
            ############

            w_up34=w_up34+nn*diff_o.T     ### update out_layer (n+1,10)
            #print w_23.shape
            motion_a=diff_o.T
            #print  (np.dot(w_23,yr.T)).shape
            ##ok
            #########
            w_d=np.array(w_up34[0:nb,:])                ### (n,10)
            #print nn
            yr2=(np.dot(w_d,yr.T)).T*diff_out_1(o_in1)   ### (n,10).(10,1)----(n,1)-->(1,n)
            diff_h=np.dot(yr2.T,)
            diff_h=np.dot(yr2.T,input_arra)             ### (n,1).(1,785)---(n,785)
            #print diff_h.shape
            #w_up23=w_up23+n*diff_o.T
            w_up12=w_up12+nn*diff_h.T   ### update  hid_layer (785,n)
            motion_b=diff_h.T
        #mm_precision=my_test(w_up12,w_up23, X_val1, y_val1)
        #print  mm_precision
            #if pi%5==0 or pi==pb-1:
                #lra=lra+ l_err(lab_out,y_re)## mse
                #sub=np.sum(abs(lab_out-y_re))
                #if sub==0:
                    #mm_pe=mm_pe+1
        """##-Five times per iteration calculation-##"""
        if pi%5==0 or pi==pb-1:
            lra,mm_pe=my_test(w_up12,w_up23, X_val1, y_val1)
            pb_mse.append(lra)
            md_precision.append(mm_pe)
            """##--print-mse--##"""
            print lra
            """##--print-accuracy--##"""
            print  mm_pe
    #mm_precision=my_test(w_up12,w_up23, X_val1, y_val1)
    #print  mm_precision
        #pi=pi+1
    ##########

    #  plt.show()
    #plt.figure(figsize=(5,5))
    #plt.plot(range(1,pb+1), pb_mse)

    #plt.show()

    return w_up12,w_up23,pb_mse,md_precision
############# --processing data(:,28*28)
def my_data(mdata):
    [m_x, m_y, m_z, m_j]=mdata.shape
    new_data=np.zeros([m_x,m_z*m_j])
    for i in range(m_x):
        new_data[i,:]=mdata[i][0].reshape((1, m_z*m_j))
    ad=np.ones([m_x, 1])
    new_data=np.c_[new_data,ad]
    return  new_data
########## main

pb=61      ## 5*x+1
#bais=0.1
X_train, y_train, X_val, y_val, X_test, y_test = im.load_dataset()
m_Xtrain=my_data(X_train)  ##### (50000,785)
m_Xval=my_data(X_val)      ##### (10000, 785)
m_Xtest=my_data(X_test)   #### (10000, 785)


mm_xtrain=m_Xtrain[0:15000,:]
mm_ytrain=y_train[0:15000]
mm_xval=m_Xval[0:5000,:]
mm_yval=y_val[0:5000]

num_a=int((pb-1)/5)+1
###################
"""
################# bais
sd=np.linspace(-1,1,50)
nb=16     ## hidden node
ww_12=np.zeros([785,nb])
ww_23=np.zeros([nb+1,10])

b_arr_mse=np.zeros([50,num_a])
b_arr_persion=np.zeros([50,num_a])
for bais in [0.3]:
    kl=0
    ww_12=weight_init(ww_12, bais)
    ww_23=weight_init(ww_23, bais)
    ##########
    #print bais
    w12,w23,mse_a,mpersion=bp(mm_xtrain, mm_ytrain, m_Xval, y_val, ww_12, ww_23, pb, nn)
    b_arr_mse[kl,:]=np.array([mse_a])
    b_arr_persion[kl, :]=np.array([mpersion])
    kl=kl+1
#plot--
"""
#####################
#################### num of hidden node
"""##LearnRate iteration##"""
nb_arr_mse=np.zeros([24,num_a])
nb_arr_persion=np.zeros([24,num_a])                                  ## --0.1  ##learn rate
bais=0.3
kl=0
for nn_learn in [0.1,0.3,0.5,0.7]:
    nn=nn_learn
    """##The number of hidden nodes iteration relation"""
    for nb in range(12,30,3):
        print nb
        ww_12=np.zeros([785,nb])
        ww_23=np.zeros([nb+1,nb])
        ww_34=np.zeros([nb+1,10])
        ww_12=weight_init(ww_12, bais)
        ww_23=weight_init(ww_23, bais)
        ww_34=weight_init(ww_34, bais)
        ##########
        w12,w23,mse_a2,mpersion2=bp(mm_xtrain, mm_ytrain, m_Xval, y_val, ww_12, ww_23,ww_34, pb, nn,nb)
        nb_arr_mse[kl,:]=np.array([mse_a2])
        nb_arr_persion[kl, :]=np.array([mpersion2])
        kl=kl+1
    ### plot
np.savetxt("mse.txt",nb_arr_mse)
np.savetxt("persion.txt",nb_arr_persion)
    #xx_1=np.zeros([nb_x,nb_y])
    #for i in range(nb_x):
        #xx_1[i,:]=np.array([xx])
    #plt.figure(figsize=(12,9))



#########--print--last iteration relation
br1=nb_arr_mse[:,-1].transpose()
print br1
pr1=nb_arr_persion[:,-1].transpose()
#plt.plot(aaa, br1,'y')
#plt.plot(aaa, pr1,'b')
print pr1
#plt.show()
###########
#########last iteration relation
