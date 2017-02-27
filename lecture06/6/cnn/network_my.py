from __future__ import division
import copy
import numpy as np
import numpy.random as random
import scipy.signal as sig
import cnn16.function_c as fun
import cnn16.layers_my as layers
import cnn16.load_and_extract_mnist_data as im
import cnn16.network as bpnet
import cnn16.processing_data as proce
import cnn16.logistic as logistic_
import cnn16.no_soigmoid as n_soigmoid_
import cnn16.relu as relu_
import cnn16.tanh as tanh_
import cnn16.classifier as classifier_
import cnn16.softmax as softmax_
class network:
    def __init__(self, input_a, val, epochs, node, num_fiters, size, choose_soigmoid):
        self.input_a=input_a
        self.val=val
        self.epochs=epochs
        self.node=node
        self.num_fiters=num_fiters
        self.size=size
        self.choose_soigmoid=choose_soigmoid
    @staticmethod
    def choose_init_soigmoid(fn, input_a):
        if fn==0:
            invoke_ac=n_soigmoid_.no_soigmoid(input_a)
        if fn==1:
            invoke_ac=logistic_.logistic(input_a)
        if fn==2:
            invoke_ac=relu_.relu(input_a)
        if fn==3:
            invoke_ac=tanh_.tanh(input_a)
        if fn==4:
            invoke_ac=classifier_.classifier(input_a)
        if fn==5:
            invoke_ac=softmax_.softmax(input_a)
        return invoke_ac
    def net(self):
        [a1, a2, a3, a4]=self.input_a.shape
        n_len=len(self.node)
        list_layers=[]
        activation_cn=[]
        for k in range(n_len):
            if self.node[k]==0:
                layer_0=layers.InputLayer(self.input_a, self.num_fiters[k], self.size[k])
                list_layers.append(layer_0)
                ac_1=network.choose_init_soigmoid(self.choose_soigmoid[0] ,self.input_a)
                activation_cn.append(ac_1)
            if self.node[k]==1:
                layer_1=layers.ConvolutionLayer(list_layers[k-1].output_a, self.num_fiters[k], self.size[k])
                layer_1.connect(list_layers[k-1])
                list_layers.append(layer_1)
                ac_2=network.choose_init_soigmoid(self.choose_soigmoid[k] ,list_layers[k].output_a)
                activation_cn.append(ac_2)
            if self.node[k]==2:
                layer_2=layers.MaxPoolingLayer(list_layers[k-1].output_a,  self.num_fiters[k], self.size[k])
                layer_2.connect(list_layers[k-1])
                list_layers.append(layer_2)
                ac_3=network.choose_init_soigmoid(self.choose_soigmoid[k] ,list_layers[k].output_a)
                activation_cn.append(ac_3)

        ##forward
        mm_xtrain, mm_ytrain, mm_xval, mm_yval=proce.main_data()
        for i in range(self.epochs):
            #print self.input_a[0:1000,].shape
            self.net_forward(self.input_a[0:1000,],list_layers,activation_cn, n_len)
            """
            list_layers[0].input_a=self.input_a[j,]
            for k in range(1,n_len):
                list_layers[k].input_a=list_layers[k-1].output_a
                list_layers[k].forward(list_layers[k-1])
                activation_cn[k].input_a=list_layers[k].output_a
            #print list_layers[k].weight.shape       ##(120,400)
            """
            list_layers_2=copy.deepcopy(list_layers)
            activation_cn_2=copy.deepcopy(activation_cn)
            #test_ar=np.zeros([120, 100])
            self.net_forward(self.input_a[0:1000,],list_layers_2,activation_cn_2, n_len)
            test_ar= list_layers_2[k].output_a
            #print list_layers_2[k].output_a.shape
            #print list_layers[k].output_a.shape
            net_2=bpnet.network([120,84,10],0.3, 0.2,[0,1,1],[5], list_layers[k].output_a, mm_ytrain[0:1000,], test_ar, mm_ytrain[0:1000,])


            #print pb_mse
            #print md_pre
            #print list_layers[k].output_a.shape
    ##backword
            ##full_connect
            [pb_mse, md_pre, yr, back_weight]=net_2.net()
            print pb_mse, md_pre
            diff_f_2=activation_cn[k].diff_de()
            diff_f_2=np.sum(diff_f_2,0)*1.0/diff_f_2.shape[0]
            #asd=np.reshape(np.dot(back_weight[:-1,:], yr.T).T, (diff_f_2.shape[0],diff_f_2.shape[1],diff_f_2.shape[2]))
            #c_f_deta=diff_f_2*asd    #(120,1)
            ##cnn
            print np.dot(back_weight[:-1,:], yr.T).shape,diff_f_2.shape
            c_f_deta=diff_f_2*np.dot(back_weight[:-1,:], yr.T)
            #c_f_deta=np.sum(c_f_deta,0)
            print c_f_deta.shape
            list_layers[n_len-1].grad=np.reshape(c_f_deta, (c_f_deta.shape[0],c_f_deta.shape[1],c_f_deta.shape[1]))
            #list_layers[n_len-1].grad=c_f_deta
            #print list_layers[n_len-1].grad.shape
            for bk in range(n_len-2, -1, -1):
                list_layers[bk].backward(list_layers[bk+1])

            #print list_layers[1].weight
            #list_layers[bk-1]
            #print list_layers[bk].grad.shape
    def net_forward(self, inputa,list_layers,activation_cn, n_len):
        list_layers[0].input_a=list_layers[0].output_a=inputa
        #print inputa.shape
        activation_cn[0].input_a=list_layers[0].output_a
        list_layers[0].output_a=activation_cn[0].activation()
        for k in range(1,n_len):
            list_layers[k].input_a=list_layers[k-1].output_a
            list_layers[k].forward(list_layers[k-1])
            activation_cn[k].input_a=list_layers[k].output_a
            list_layers[k].output_a=activation_cn[k].activation()




        #return list_layers
if __name__=="__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = im.load_dataset()
    net_1=network(input_a=X_train, val=X_val,epochs=10, node=[0, 1, 2, 1, 2, 1], num_fiters=[1, 6, 6, 16 ,16, 120],
                  size=[28, 5, 2, 5, 2, 4], choose_soigmoid=[0,0,0,0,0,0])
    my_layers=net_1.net()
                                                #node 0:input_layer 1:ConvolutionLayer 2:MaxPoolingLayer
    #last_layer=my_layers[len(net_1.node)-1]        #fn==0:no_soigmoid
                                                    #fn==1:logistic
                                                    #fn==2:relu
                                                    #fn==3:tanh
                                                    #fn==4:classifier
                                                    #fn==5: softmax
