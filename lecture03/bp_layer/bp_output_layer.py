from __future__ import division
import numpy as np
import random
import matplotlib.pylab as plt
import math
import bp_layer.bp_top as bp
class output_layer(bp.bp_layer):
    def __init__(self, input_a, output_a):
        bp.bp_layer.__init__(self, input_a, output_a)
        #self.forward_weight=forward_weight
    def activation(self):
        s=1.0/(1+np.exp(-1*self.input_a))
        return s
    def classifier(self):
        [v_x,v_y]=self.output_a.shape
        d=np.zeros([v_x,v_y])
        for i in range(v_y):
            if self.output_a[0,i]<0.7:
                d[0,i]=0
            if self.output_a[0,i]>0.7:
                d[0,i]=1
        return  d
    def softmax(self):
        [v_x,v_y]=self.output_a.shape
        d=np.zeros([v_x,v_y])
        max_=np.max(self.output_a)
        out_copy=np.exp(self.output_a-max_)
        sum_=np.sum(out_copy)
        out_copy=out_copy*1.0/sum_
        [ax,ay]=np.where(out_copy==np.max(out_copy))
        d[ax,ay]=1
        return  d
    def diff_out_1(self, y, y_r):
        g=(1-self.activation())*self.activation()*(y_r-y)
        return g