import bp_layer.processing_data as proce
import network as net_
import numpy as np
import matplotlib.pylab as plt
if __name__=="__main__":
    mm_xtrain, mm_ytrain, mm_xval, mm_yval=proce.main_data()
    net_1=net_.network([784,15,10],0.3, 0.5,61, mm_xtrain, mm_ytrain, mm_xval, mm_yval)
    [pb_mse, md_pre]=net_1.net()
