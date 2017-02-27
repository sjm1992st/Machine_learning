import numpy as np
import random
import math
import matplotlib.pylab as plt
from scipy import stats
#######
def uniform(u_num):
    list_a = []
    for i in range(u_num):
        list_a.append(random.uniform(-1, 1.0000001))
    return list_a
######
def Gaussian(G_num):
        list_a = []
        while(G_num>0):
            u = random.uniform(0, 1.0000001)
            v = random.uniform(0, 1.0000001)
            if u <1 :
                s = -1
            if u==1:
                s =1
            # print Gy
            if 0<u<=1:
                Gx =  math.sqrt(s*2 * math.log(u)) * math.sin(2 * math.pi*v)
                list_a.append(Gx)
                G_num=G_num-1
        return list_a
    #####
def Bernoulli(B_num):
    list_a = []
    for i in range(B_num):
        ax=random.uniform(0, 1)
        if ax<0.3:
            y=1
        if ax>=0.3:
            y=0
        list_a.append(y)
    return list_a
####################
####### central_limit
def centeral(cout):
    list_sum=[]
    for num_r in range(cout):
        sum=0
        for num_c in range(cout):
            coin = random.uniform(0, 1)
            if coin<0.5:
                x_f=-1
            if coin>=0.5:
                x_f=1
            sum=sum+x_f
        list_sum.append(sum)
    return list_sum
###################
###################
##--main
i=1
mean_a=[]
mean_b=[]
mean_c=[]
plt.figure(figsize=(18,8))
for num in [10,30,100,300,1000]:
    list_a=uniform(num)
    list_b=Gaussian(num)
    list_c=Bernoulli(num)
    plt.subplot(5,3,i)
    plt.hist(list_a,bins=65)
    plt.subplot(5,3,i+1)
    plt.hist(list_b,bins=65)
    plt.subplot(5,3,i+2)
    plt.hist(list_c)
    i=i+3
    #mean_a.append(np.mean(list_a))
    #mean_b.append(np.mean(list_b))
    #mean_c.append(np.mean(list_c))
    #print np.mean(list_a)
    #print np.var(list_a)
plt.show()
#print list_a
##################  law of large numbers
plt.figure(figsize=(11,6))
xx=range(10,1000,15)
for ad in xx:
    list_a2=uniform(ad)
    list_b2=Gaussian(ad)
    list_c2=Bernoulli(ad)
    mean_a.append(np.mean(list_a2))
    mean_b.append(np.mean(list_b2))
    mean_c.append(np.mean(list_c2))
#print mean_a
#print mean_b
#print mean_c
plt.plot(xx,mean_a,'r',marker='o',label='uniform')
plt.plot(xx,mean_b,'g',marker='*',label='Gaussian')
plt.plot(xx,mean_c,'b',marker='+',label='Bernoulli')
#plt.axis([10,1000,0,1])
plt.legend()
plt.show()
###############

######## central_limit
cou=1000
nm=31    ## Divided into nm
ran=centeral(cou)########---mean=0,var=1
plt.hist(ran,bins=nm,color='c')### ~~~N(0,1000).
d=np.linspace(-100,100,nm+1)
id=0
dy=[]
f=[]
while(id<nm):###### subsection integral
    dy.append((stats.norm.cdf(d[id+1],0,math.sqrt(cou))-stats.norm.cdf(d[id],0,math.sqrt(cou)))*cou)
    f.append((d[id+1]+d[id])*0.5)
    id=id+1
print sum(dy) #1000
plt.plot(f,dy,'r')
plt.title("central_limit")
plt.show()








