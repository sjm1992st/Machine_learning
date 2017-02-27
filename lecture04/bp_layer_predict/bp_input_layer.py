from __future__ import division
import numpy as np
import random
import matplotlib.pylab as plt
import math
import bp_layer_predict.bp_top as bp
class input_layer(bp.bp_layer):
    def __init__(self, input_a, output_a, back_weight):
        bp.bp_layer.__init__(self, input_a, output_a)
        self.back_weight=back_weight
    def next_in(self):
        h_in=np.dot(self.output_a, self.back_weight)
        return h_in