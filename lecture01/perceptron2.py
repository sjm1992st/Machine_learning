import numpy as np
import random
import matplotlib.pylab as plt
import math


def dy(y_pr, array_a):
    if sum(y_pr) >= 0:
        y_value = 1
    if sum(y_pr) < 0:
        y_value = -1
    if array_a == 1:
        d = 1
    if array_a == 2:
        d = -1
    return d - y_value

def average(array_a, array_weight,z):
    sum_dy=0
    for h in range(z):
        y_pr3 = array_a[h, 0:3] * array_weight[0, :]
        d_y3 = dy(y_pr3, array_a[h, 3])
        sum_dy=sum_dy + d_y3
    return  sum_dy

def perceptron(r, w, d, na, nb):
    array_a = np.zeros((na + nb, 4))
    #na_copy = na
    #nb_cpoy = nb
    i = 0
    # j = 0
    while (na > 0):
        ax = random.uniform(-r - 0.5 * w, r + 0.5 * w)
        ay = random.uniform(0, r + 0.5 * w)
        if (r - 0.5 * w) ** 2 <= ax ** 2 + ay ** 2 <= (r + 0.5 * w) ** 2:
            array_a[i, 0] = 1
            array_a[i, 1] = ax
            array_a[i, 2] = ay
            # print mylist_a[i]
            array_a[i, 3] = 1  ##lable
            i = i + 1
            na = na - 1
            #
    while (nb > 0):
        bx = random.uniform(- 0.5 * w, 2 * r + 0.5 * w)
        by = random.uniform(- (d + 0.5 * w + r), - d)
        if (r - 0.5 * w) ** 2 <= (bx - r) ** 2 + (by + d) ** 2 <= (r + 0.5 * w) ** 2:
            array_a[i, 0] = 1
            array_a[i, 1] = bx
            array_a[i, 2] = by
            array_a[i, 3] = 2  # lable
            i = i + 1
            nb = nb - 1
    return array_a
##pc



def calculation(r,w,na_copy, nb_copy,array_a,array_weight,pb,b):
    #array_weight = np.zeros((1, 3))
    seq = range(na_copy + nb_copy)
    random.shuffle(seq)
    # for k in range(na+nb)
    #array_weight[0, 0] = 0
    pi = 0
    #pb = 2000
    pb_mse = []
    while( pi < pb):
        sum_mse = 0
        for k in seq:
            # array_weight[0, 0] = 1
            y_pr = array_a[k, 0:3] * array_weight[0, :]
            d_y = dy(y_pr, array_a[k, 3])
            p = (10 ** -5 - 10 ** -1) * 1.0 / (na_copy + nb_copy - 1) * pi + 10 ** -1
            sum_mse = sum_mse + d_y ** 2
            array_weight = array_weight + p* (d_y) * array_a[k, 0:3]
            array_weight[0, 0] = b
            # print d_y2
            # print sum_mse
        mse_q = sum_mse * 1.0 / (na_copy + nb_copy)
            # print mse_q
        pb_mse.append(math.sqrt(mse_q))
        pi = pi + 1
       # array_weight = array_weight + p * (d_y) * array_a[k, 0:3]
    #print  array_weight
    return array_weight,pb_mse

# print p


##plot
def plot(r,w,na_copy,nb_copy,array_a,array_weight,pb,pb_mse):
    al = int(math.floor(-r - 0.5 * w))
    ad = int(math.ceil(2 * r + 0.5 * w))
    wx = np.array(range(al, ad))
# print wx
    if array_weight[0, 2]!=0:
        wy = -array_weight[0, 1] * 1.0 / array_weight[0, 2] * wx - array_weight[0, 0] * 1.0 / array_weight[0, 2]
    if array_weight[0, 2]==0:
        wy=wx*0
# print wy
    plt.plot(array_a[0:na_copy, 1], array_a[0:na_copy, 2], '+r', array_a[na_copy:nb_copy + na_copy, 1],
         array_a[na_copy:nb_copy + na_copy, 2], '+b')
    plt.plot(wx, wy, 'y')
    plt.show()
   # print  pb_mse
    #print range(pb)
    plt.plot(range(pb), pb_mse)
    plt.show()


##main
r=10
w=6
d=1
pb=50
b=1
na=1000
nb=1000
array_a=perceptron(r, w, d, na, nb)
array_weight = np.zeros((1, 3))
array_weight[0, 0] = b
array_weight_train,pb_mse_train=calculation(r,w,na,nb,array_a,array_weight,pb,b)
#array_test=perceptron(10, 6, 1, 2000, 2000)
print array_weight_train
print pb_mse_train
plot(r,w,na,nb,array_a,array_weight_train,pb,pb_mse_train)
#array_weight_train[0,0]=b
#array_weight_test,pb_mse_test=calculation(r,w,2000,2000,array_test,array_weight_train,pb)
#print array_weight_test
#plot(r,w,2000,2000,array_test,array_weight_test,pb,pb_mse_test)
