import random
import math
import matplotlib as plot
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pylab as plt

def creat_set(num_z, bool_def, flag_noise=0):
    arra = np.zeros([num_z, 2])
    k = 0
    x_set=np.linspace(-3,3,num_z)
    while (num_z > 0):
        u = random.uniform(0, 1.0000001)
        v = random.uniform(0, 1.0000001)
        if u < 1:
            s = -1
        if u == 1:
            s = 1
        # print Gy
        # print Gy * (2 * math.pi) ** 0.5
        # print  math.log(Gy * (2 * math.pi) ** 0.5)
        if 0 < u <= 1:
            x=x_set[num_z-1]
            Gx = math.sqrt(s * 2 * math.log(u)) * math.sin(2 * math.pi * v)
            if flag_noise==1:
                x=x+Gx*math.sqrt(0.2)
            if bool_def == 1:
                y = 2 * x + 1 + Gx
            if bool_def == 2:
                y = 0.01 * x ** 2 + 2 * x + 1 + Gx
            if bool_def == 3:
                y = 0.1 * math.exp(0.1 * x) + 2 * x + 1 + Gx
            arra[k, 0] = x
            arra[k, 1] = y
            num_z = num_z - 1
            k = k + 1
    return arra


def L_S_1(x):    #solve_model_1
    a=float(x[0])
    b=float(x[1])
    lam1=0
    lam2=0
    lam3=0
    lam4=0
    for i in range(num):
        lam1=lam1+arra[i,0]**2
        lam2=lam2+arra[i,0]
        lam3=lam3+arra[i,0]*arra[i,1]
        lam4=lam4+arra[i,1]
    return [
            a*lam1+b*lam2-lam3,
            a*lam2+num*b-lam4
        ]


def L_S_2(x):    #solve_model_2
    a=float(x[0])
    b=float(x[1])
    c=float(x[2])
    lam1=0
    lam2=0
    lam3=0
    lam4=0
    lam5=0
    lam6=0
    lam7=0
    for i in range(num):
        lam1=lam1+arra[i,0]**2
        lam2=lam2+arra[i,0]
        lam3=lam3+arra[i,0]*arra[i,1]
        lam4=lam4+arra[i,1]
        lam5=lam5+arra[i,0]**4
        lam6=lam6+arra[i,0]**3
        lam7=lam7+arra[i,1]*arra[i,0]**2
    return [
            a*lam5+b*lam6+c*lam1-lam7,
            a*lam6+b*lam1+c*lam2-lam3,
            a*lam1+b*lam2+c*num-lam4
        ]


def L_S_3(x):    #solve_model_3
    a=float(x[0])
    b=float(x[1])
    c=float(x[2])
    d=float(x[3])
    lam1=0
    lam2=0
    lam3=0
    lam4=0
    lam5=0
    lam6=0
    lam7=0
    lam8=0
    lam9=0
    lam10=0
    for i in range(num):
        lam1=lam1+arra[i,0]**2
        lam2=lam2+arra[i,0]
        lam3=lam3+arra[i,0]*arra[i,1]
        lam4=lam4+arra[i,1]
        lam5=lam5+arra[i,0]**4
        lam6=lam6+arra[i,0]**3
        lam7=lam7+arra[i,1]*arra[i,0]**2
        lam8=lam8+arra[i,0]**5
        lam9=lam9+arra[i,0]**6
        lam10=lam10+arra[i,1]*arra[i,0]**3
    return [
            a*lam9+b*lam8+c*lam5+d*lam6-lam10,
            a*lam8+b*lam5+c*lam6+d*lam1-lam7,
            a*lam5+b*lam6+c*lam1+d*lam2-lam3,
            a*lam6+b*lam1+c*lam2+d*num-lam4
        ]


def plot_model(arra,flag_model):    ###plot_model
    plt.plot(arra[:,0],arra[:,1],'r*')
    y1=[]
    xz=np.linspace(-4,4,30)
    if flag_model==1:
        result = fsolve(L_S_1,[1, 1])
        for az in xz:
            y1.append(result[0]*az+result[1])
    if flag_model==2:
        result = fsolve(L_S_2,[1, 1, 1])
        for az in xz:
            y1.append(result[0]*az**2+result[1]*az+result[2])
    if flag_model==3:
        result = fsolve(L_S_3,[1, 1, 1, 1])
        for az in xz:
            y1.append(result[0]*az**3+result[1]*az**2+result[2]*az+result[3])
    plt.plot(xz,y1)
    #plt.show()
    print result


### main
num=100
#### y-noise
jk_1=0
plt.figure(figsize=(13,11))
for def_n in range(1,4):
    arra=creat_set(100, def_n)
    for plot_n in range(1,4):
        jk_1=jk_1+1
        plt.subplot(3,3,jk_1)
        plot_model(arra,plot_n)
        if jk_1==1:
            plt.title("$y=2x+1--1$")
        if jk_1==2:
            plt.title("$y=2x+1--2$")
        if jk_1==3:
            plt.title("$y=2x+1--3$")
        if jk_1==4:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---1$")
        if jk_1==5:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---2$")
        if jk_1==6:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---3$")
        if jk_1==7:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---1$")
        if jk_1==8:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---2$")
        if jk_1==9:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---3$")
plt.show()
#### x,y-noise
jk_2=0
plt.figure(figsize=(13,11))
for def_n in range(1,4):
    arra=creat_set(100, def_n, 1)
    for plot_n in range(1,4):
        jk_2=jk_2+1
        plt.subplot(3,3,jk_2)
        plot_model(arra,plot_n)
        if jk_2==1:
            plt.title("$y=2x+1--1$")
        if jk_2==2:
            plt.title("$y=2x+1--2$")
        if jk_2==3:
            plt.title("$y=2x+1--3$")
        if jk_2==4:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---1$")
        if jk_2==5:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---2$")
        if jk_2==6:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---3$")
        if jk_2==7:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---1$")
        if jk_2==8:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---2$")
        if jk_2==9:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---3$")
plt.show()
        #plt.show()

######## add(0,3)
jk_3=0
plt.figure(figsize=(13,11))
num=100+1
for def_n in range(1,4):
    arra=creat_set(100, def_n)
    x_1=np.array([[0,3]])
    arra=np.r_[arra,x_1]
    for plot_n in range(1,4):
        jk_3=jk_3+1
        plt.subplot(3,3,jk_3)
        plot_model(arra,plot_n)
        if jk_3==1:
            plt.title("$y=2x+1--1$")
        if jk_3==2:
            plt.title("$y=2x+1--2$")
        if jk_3==3:
            plt.title("$y=2x+1--3$")
        if jk_3==4:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---1$")
        if jk_3==5:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---2$")
        if jk_3==6:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---3$")
        if jk_3==7:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---1$")
        if jk_3==8:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---2$")
        if jk_3==9:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---3$")
plt.show()
        #plt.show()
#######
###### add(4,0)
jk_4=0
plt.figure(figsize=(13,11))
num=100+1
for def_n in range(1,4):
    arra=creat_set(100, def_n)
    x_2=np.array([[4,0]])
    arra=np.r_[arra,x_2]
    for plot_n in range(1,4):
        jk_4=jk_4+1
        plt.subplot(3,3,jk_4)
        plot_model(arra,plot_n)
        if jk_4==1:
            plt.title("$y=2x+1--1$")
        if jk_4==2:
            plt.title("$y=2x+1--2$")
        if jk_4==3:
            plt.title("$y=2x+1--3$")
        if jk_4==4:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---1$")
        if jk_4==5:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---2$")
        if jk_4==6:
            plt.title("$0.01x^{2}+2x+1+\\varepsilon---3$")
        if jk_4==7:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---1$")
        if jk_4==8:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---2$")
        if jk_4==9:
            plt.title("$0.1e^{0.1x}+2x+1+\\varepsilon---3$")
plt.show()
        #plt.show()
######

