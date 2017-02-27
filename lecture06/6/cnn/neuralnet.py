from __future__ import division
from sklearn.cross_validation import train_test_split
import cnn16.processing_data as proce
import load_and_extract_mnist_data as im
from scipy.signal import convolve2d, correlate2d
from layers import InputLayer, FullyConnectedLayer, ReLuLayer,\
                   ConvolutionLayer, PoolingLayer, SquaredLossLayer, SoftmaxLossLayer

import numpy as np
#import pp
import datetime
import math, copy, time

class NeuralNet:
    def __init__(self, layers, decay=0.001, learning_rate=0.1):
        mapping = {"input": lambda x: InputLayer(x),
                   "fc": lambda x: FullyConnectedLayer(x),
                   "convolution": lambda x: ConvolutionLayer(x),
                   "pool": lambda x: PoolingLayer(x),
                   "squaredloss": lambda x: SquaredLossLayer(x),
                   "softmax": lambda x: SoftmaxLossLayer(x),
                   "relu": lambda x: ReLuLayer(x),}

        self.layers = []
        self.decay = decay

        self.learning_rate = learning_rate
        prev_layer = None

        for layer in layers:
            layer["input_shape"] = layer.get("input_shape", None) or prev_layer.output_shape
            layer["decay"] = self.decay
            layer = mapping[layer["type"]](layer)
            self.layers.append(layer)
            prev_layer = layer

    def forward(self, input):
        inputs = [input]

        for layer in self.layers:

            inputs.append(layer.forward(inputs[-1])) #Each layer of the input
        return inputs

    def backward(self, inputs, parent_gradient):
        gradients = [parent_gradient]

        for input, layer in zip(inputs[:-1][::-1], self.layers[::-1]):   ##invert of list and lose the last
            #print
            gradients.append(layer.backward(input, gradients[-1]))

        return gradients

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update_weight(rate)

    def loss(self, input, expected, layers):
        prediction =self.predict(input, layers)

        loss = layers[-1].loss(prediction, expected)
        """
        for layer in layers[:-1][::-1]:
            loss += layer.loss() # regularization terms
        """
        return loss

    def predict(self, buffer):
        inputs = [buffer]
        #copy_layers=copy.deepcopy(layers)
        for layer in self.layers:

            inputs.append(layer.predict(inputs[-1]))

        return inputs[-1]

    def train(self, X, y, n_epochs, batch_size):
        j=0
        list_loss=[]
        indices = np.arange(len(X))
        np.random.shuffle(indices)   #Disrupted sample order
        for epoch in range(0, n_epochs):
            #for j in range(0,int(len(X)/batch_size)):
            if j>len(X):
                j=0
            X_bath=X[indices[j:j+batch_size]]
            y_bath=y[indices[j:j+batch_size]]
            j=j+batch_size
            start_f = time.clock()
            #starttime = datetime.datetime.now()
            inputs = self.forward(X_bath)
            end_f = time.clock()
            J_loss=self.layers[-1].loss(inputs[-1], y_bath)
            start_b = time.clock()
            gradients = self.backward(inputs, y_bath)
            end_b = time.clock()
            self.update_weight(self.learning_rate)
            #end = time.clock()
            #endtime = datetime.datetime.now()
            """ ##check__gradient:
            if epoch==0:
                res=self.check__gradient(X_bath, y_bath)
                print "check__gradient: %d" %(res)
            """
            list_loss.append(J_loss)
            if epoch%50==0:
                print "epoch:%d batch_size:%d forward %.3f seconds/backward %.3f seconds"\
                      %(epoch+1,batch_size,(end_f-start_f), ((end_b-start_b)))
                #print "%d time: %f seconds" %((endtime - starttime).seconds)
                right_rate=self.test_error(test_data, test_target)
                print "pre:{:.2f}%".format(right_rate)
                print "learn_rate:%f" %(self.learning_rate)
                print "loss: %f" %(J_loss)
            """
                if epoch>99 and abs((sum(list_loss[-100:-50])-sum(list_loss[-50:])))*1.0/50<0.1:
                    self.learning_rate=self.learning_rate*1.0/2
            """
            if epoch>199 and epoch%200==0:
                self.learning_rate=self.learning_rate*1.0*0.3

            #print "loss: %f" %(J_loss)
                #self.learning_rate=0.04*(1-right_rate*1.0/100)*100000/100000


    def check__gradient(self, input, expected):
        #print input.shape
        eps = 10**-7
        pert = input.copy()
        index_layer=1
        res = np.zeros(self.layers[index_layer].dfilter.shape)
        copy_1=copy.deepcopy(self.layers)
        copy_2=copy.deepcopy(self.layers)
        [a1,a2,a3,a4]=copy_1[index_layer].filter.shape
        for m in range(a1):
            for n in range(a2):
                for i in range(a3):
                    for j in range(a4):
                        copy_1[index_layer].filter[m,n,i,j]=copy_1[1].filter[m,n,i,j]+eps
                        J1=self.loss(pert, expected,copy_1)
                        J2=self.loss(input, expected, copy_2)
                        res[m,n,i,j]=(J1 - J2)*1.0/eps
                        copy_1[index_layer].filter[m,n,i,j]=copy_1[1].filter[m,n,i,j]-eps
        inputs = self.forward(input)
        gradients = self.backward(inputs, expected)
        #gradients_a=self.backward(a2[1], expected)
        #print res
        #print self.layers[1].dfilter
        """
        for index, x in np.ndenumerate(input):
            pert[index] = input[index] + eps
            res[index] = (self.loss(pert, expected) - self.loss(input, expected))/eps
            pert[index] = input[index]
        """
        return np.sqrt(np.sum(res-self.layers[index_layer].dfilter)**2)

    #@staticmethod
    def test_error(self, X, y):
        assert(len(X) == len(y))
        zy_numer=net.predict(X)
        tran_y=np.zeros(zy_numer.shape)
        #print zy_numer[0,:]
        #print np.max(zy_numer[0,:])
        for i in range(len(zy_numer)):
            tran_y[i,np.argmax(zy_numer[i,:])]=1
        #print tran_y[0,:]
        #print y[0,:]
        #print y.shape
        #print tran_y.shape
        zx=np.sum(abs(tran_y- y),1)
        #print zx.shape
        #zx=[zx]
        #print zx
        zy=np.where(zx >0, 1, 0)
        #print zy
        return 100*(1-np.sum(zy)*1.0/len(X))

if __name__ == "__main__":
    n_classes = 10
    net = NeuralNet([{"type": "input", "input_shape": (1, 28, 28)},
                     {"type": "convolution", "filters": 20, "size": 5},
                     {"type": "relu"},
                     {"type": "pool", "size": 2},
                     {"type": "convolution", "filters": 50, "size": 5},
                     {"type": "relu"},
                     {"type": "pool", "size": 2},
                     {"type": "fc", "neurons": 120},
                     #{"type": "convolution", "filters": 120, "size": 4},
                     {"type": "relu"},
                     {"type": "fc", "neurons": 500},
                     {"type": "relu"},
                     {"type": "fc", "neurons": n_classes},
                     {"type": "relu"},
                     {"type": "softmax", "categories": n_classes}])

    X_train, y_train, X_val, y_val, X_test, y_test = im.load_dataset()
    mm_xtrain, mm_ytrain, mm_xval, mm_yval=proce.main_data()

    train_data, test_data, train_target, test_target = train_test_split(mm_xtrain[0:10000,], mm_ytrain[0:10000], train_size=0.7)
    train_data = train_data.reshape((len(train_data),1, 28, 28))
    test_data = test_data.reshape((len(test_data), 1, 28, 28))
    """
    train_data=X_train
    test_data=X_val
    train_target=mm_ytrain
    test_target=mm_yval
    """
    #print test_data.shape, test_target.shape
    net.train(train_data, train_target, n_epochs=1001, batch_size=64)
    """
    ###Parallel python cpu
    ppservers = ()
    #ppservers = ("10.0.0.1",)
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
        # Creates jobserver with ncpus workers
        job_server = pp.Server(ncpus, ppservers=ppservers)
    else:
        # Creates jobserver with automatically detected number of workers
        job_server = pp.Server(ppservers=ppservers)
    print "pp ", job_server.get_ncpus(), "workers"
    job_server.print_stats()
    """

