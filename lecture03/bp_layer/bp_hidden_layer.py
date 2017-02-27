from __future__ import division
import numpy as np
import random
import matplotlib.pylab as plt
import math
import bp_layer.bp_top as bp
class hidden_layer(bp.bp_layer):
    def __init__(self, input_a, output_a, back_weight):
        bp.bp_layer.__init__(self, input_a, output_a)
        #self.forward_weight=forward_weight
        self.back_weight=back_weight
    def next_in(self):
        h_hid=np.dot(self.output_a, self.back_weight)
        return h_hid
    def activation(self):
        s=1.0/(1+np.exp(-1*self.input_a))
        return s
    def diff_out_1(self, yr):
        w_d=self.back_weight[:-1,:]
        g=(1-self.activation())*self.activation()*(np.dot(w_d,yr.T)).T
        return g


###---output_a=hidden_layer.activation(self)












