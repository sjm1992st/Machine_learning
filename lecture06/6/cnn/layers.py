from __future__ import division
import scipy.signal as sig

import numpy as np
import numpy.random as random

class Layer:
    def __init__(self, config):

        self.input_shape = config["input_shape"]

        self.decay = config["decay"]

    def forward(self, buffer):
        return buffer

    def backward(self, input, buffer):
        return buffer

    def update_weight(self, rate):
        pass

    def predict(self, input):
        return self.forward(input)

    def loss(self):
        return 0

class InputLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        self.output_shape = self.input_shape

class FullyConnectedLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)

        assert(config["neurons"])
        neurons = config["neurons"]
        self.neurons = neurons
        self.output_shape = (neurons, )

        size=(neurons, np.prod(self.input_shape))
        self.weights = np.c_[random.uniform(-0.1,0.1, size), np.zeros(neurons)]

    def forward(self, input, weights=None):
        if weights is not None:
            weights = weights   ##(node.num*(pre_node+1))
        else :
            weights=self.weights # allow to overwrite weights for testing purposes
        an1=np.reshape(input, (len(input), np.prod(input.shape)/input.shape[0]))

        an1=np.c_[an1, np.ones([an1.shape[0],1])]
        #print (np.dot(weights, an1.T).T).shape
        return np.dot(weights, an1.T).T    #(84,865),(865,n) #node84

    def backward(self, input, parent_gradient):

        #a2=input.reshape(-1)
        #an=np.tile(input.reshape(-1), (self.neurons, 1))    #(10,84) #node10
        input=np.reshape(input, (len(input), np.prod(input.shape)/input.shape[0]))
        an2=np.c_[input, np.ones([ len(input),1])]
        #self.dweights = np.c_[input, np.ones(self.neurons)]    #(10,85)
        self.dweights= np.dot(an2.T, parent_gradient)*1.0/len(input)
        an=parent_gradient.dot(self.weights)  #(1,10)(10,85)
        #print self.dweights.shape
        return an[:,:-1]        #the

        """
        an=np.c_[input, np.ones([ len(input),1])]
        self.dweights= parent_gradient*np.dot(an, self.weights.T)
        return parent_gradient
        """
    def update_weight(self, rate):

        self.weights = self.weights - self.dweights.T*rate

    def loss(self):
        return self.decay*(np.square(self.weights.reshape(-1)[:-1])).sum()/2

    def check_gradient(self, input, params):
        eps = 10**-7
        pert = params.copy()
        res = np.zeros(shape=params.shape)

        for index, x in np.ndenumerate(params):
            neuron = index[0]
            pert[index] = params[index] + eps
            res[index] = (self.forward(input, pert)[neuron] - self.forward(input, params)[neuron])/eps
            pert[index] = params[index]

        return res

class ReLuLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        self.output_shape = self.input_shape

    def forward(self, buffer):

        ld=buffer.max()
        #print np.where(buffer < 0, 0.01, buffer*1.0/ld)
        return np.where(buffer < 0, 0.01*buffer, buffer*1.0)

    def backward(self, input, buffer):
        #dad=np.where(input < 0, 0.01, 1.0)*buffer
        return np.where(input < 0, 0.01, 1.0)*buffer

class ConvolutionLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)

        self.size = config["size"]

        self.n_filters = config["filters"]
        """
        if len(self.input_shape) == 2:
            self.n_input_maps = 1
        else :
        """
        self.n_input_maps=self.input_shape[0]
        self.output_shape = (self.n_filters, self.input_shape[-2] - self.size + 1, self.input_shape[-1] - self.size + 1)

        sz=(self.n_filters, self.n_input_maps*self.size*self.size)
        self.filter =np.random.uniform(-0.1, 0.1,size=sz)


    def forward(self, imgs):
        imgs = imgs.reshape(len(imgs),self.n_input_maps, imgs.shape[-2], imgs.shape[-1]) #4d

        self.width=imgs.shape[-2] - self.size+ 1
        self.height=imgs.shape[-1] - self.size+1
        self.output = np.zeros((len(imgs),self.filter.shape[0], self.width, self.height)) #4d

        self.filter=np.reshape(self.filter, (self.n_filters, self.n_input_maps*self.size*self.size))
        #print self.filter.shape
        feature=np.zeros([self.n_input_maps*self.size*self.size, self.width*self.height*len(imgs)]) #i*self.width*self.height---index of images
        for jk in range(len(imgs)):               ###(n_filtes*size*size)*(((M-m+1)**2)*n_img)
            for kl in range(self.n_input_maps):
                n=jk*self.width*self.height
                for j in range(self.width):
                    for k in range(self.height):
                        pad=imgs[jk,kl,j:j+self.size,k:k+self.size]
                        pad1=np.reshape(pad,(self.size*self.size,1))
                        feature[kl*(self.size*self.size):(kl+1)*self.size*self.size, n]=pad1[:,0]
                        n=n+1
        self.output =np.dot(self.filter, feature)  ##( self.n_filters, self.width*self.height*len(imgs))

        self.filter=np.reshape(self.filter, (self.n_filters,self.n_input_maps ,self.size, self.size))
        #self.filter=np.sum(self.filter, 1)*1.0/self.n_input_maps
        self.output = np.reshape(self.output, (self.filter.shape[0],len(imgs), self.width, self.height))
        self.output  = np.transpose(self.output , (1, 0, 2, 3)) #4d
        return self.output

    def backward(self, imgs, parents_gradient):
        imgs = imgs.reshape(len(imgs),self.n_input_maps, imgs.shape[-2], imgs.shape[-1])
        #print parents_gradient.shape
        input_gradient, dfilter = self.up_gradient(imgs, self.filter, parents_gradient)
        self.dfilter = dfilter
        return input_gradient

    def up_gradient(self, imgs, filters, parents_gradient):
        #if len(parents_gradient)==2:
            #parents_gradient=np.reshape(parents_gradient, (len(parents_gradient), parents_gradient.shape[1],1,1))


        #filters=np.sum(filters, 1)*1.0/self.n_input_maps
        #print filters.shape
        #imgs2_gradient = np.zeros(self.output.shape)

        imgs_gradient = np.zeros(imgs.shape)
        filters_gradient = np.zeros(filters.shape)

        for j in range(filters.shape[0]):
            for k in range(self.n_input_maps):
                for i in range(0, imgs.shape[0]):
                    r_90=np.rot90(filters[j,k,:,:])
                    r_180=np.rot90(r_90)
                    imgs_gradient[i,k,:,:]=imgs_gradient[i,k,:,:]+sig.convolve2d(parents_gradient[i,j,:,:],r_180,mode="full")
                    filters_gradient[j,k,:,:]=filters_gradient[j,k,:,:]+sig.convolve2d(imgs[i,k,:,:],parents_gradient[i,j,:,:],mode="valid")
        #filters_gradient[j,k,:,:]=filters_gradient[j,:,:]*1.0/imgs.shape[0]
        #filter_gradient += self.decay*filter
        #print filters_gradient/imgs.shape[0]
        return (imgs_gradient, filters_gradient/imgs.shape[0])

    def update_weight(self, rate):

        #new_dfilter=np.zeros(self.filter.shape)
        #for i in range(self.filter.shape[1]):
            #new_dfilter[:,i,:,:]=self.dfilter
        self.filter = self.filter - self.dfilter*rate
        #print self.filter.shape

    def loss(self):
        return self.decay*(np.square(self.filter.reshape(-1))).sum()/2

class PoolingLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        #assert(config["size"] > 0)
        #assert(len(self.input_shape) == 3)

        self.size = config["size"]
        self.output_shape = (self.input_shape[0],
                            (self.input_shape[1] - self.size)//self.size + 1,
                            (self.input_shape[2] - self.size)//self.size + 1)

    def forward(self, imgs):
        #assert(imgs.ndim == 3)
        self.maps = np.zeros([len(imgs),self.output_shape[0], self.output_shape[1], self.output_shape[2]])
        for j in range(len(imgs)):
            for i in range(0, imgs.shape[1]):
                for x in range(0, self.output_shape[-1]):
                    x_img = x*self.size
                    for y in range(0, self.output_shape[-2]):
                        y_img = y*self.size
                        self.maps[j,i,y,x] = np.max(imgs[j,i,y_img:y_img+self.size, x_img:x_img+self.size])

        return self.maps   ##4d

    def backward(self, imgs, parents_gradient):
        imgs_gradient = np.zeros(imgs.shape)
        parents_gradient=np.reshape(parents_gradient,self.maps.shape)
        for i in range(0, imgs.shape[0]):
            #img = imgs[i,]
            #img_gradient_1 = imgs_gradient[i]
            #parent_gradient = parents_gradient[i,]
            for j in range(imgs.shape[1]):

                for x in range(0, self.output_shape[1]):
                    x_img = x*self.size

                    for y in range(0, self.output_shape[2]):
                        y_img = y*self.size

                        sub = imgs[i,j, x_img:x_img+self.size,y_img:y_img+self.size]
                        sub_max_index = np.unravel_index(sub.argmax(), sub.shape)
                        #max_index = np.add(sub_max_index, (y_img, x_img))
                        pp=np.zeros([self.size,self.size])
                        pp[sub_max_index[0], sub_max_index[1]]=parents_gradient[i, j, x, y]
                        imgs_gradient[i,j,x_img:x_img+self.size,y_img:y_img+self.size]=pp
                        #ddd=parents_gradient[y, x]
                        #img_gradient_1[tuple(max_index)] = parent_gradient[y, x]

        return imgs_gradient        ##return conv_layer_deta

class SquaredLossLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)

    def forward(self, buffer):
        return buffer

    def backward(self, input, expected):
        if np.isscalar(expected):
            expected = np.array([expected])

        assert(input.shape == expected.shape)
        return input - expected

    def loss(self, predicted, expected):
        if np.isscalar(expected):
            expected = np.array([expected])
        assert(predicted.shape == expected.shape)
        return np.square(predicted - expected).sum()*0.5

class SoftmaxLossLayer(Layer):
    def __init__(self, config):
        Layer.__init__(self, config)
        self.categories = config["categories"]

    def forward(self, buffer):
        max = np.max(buffer, 1)
        max= np.reshape([max], (len(max), 1))
        #print max.shape
        exp = np.exp(buffer - max) # numerically stable
        total = np.sum(exp, 1)
        total =  np.reshape([total], (len(total), 1))
        #self.softmax=exp*1.0/total
        return 1.0*exp/total                ##(n,10)

    def backward(self, input, expected):

        output = self.forward(input)
        w=expected-output
        output2=w*input
        output3=np.sum(output2,0)     #(1,10)
        return -output3*1.0/len(output3)

    def loss(self, predicted, expected):
        #print predicted.shape
        output = predicted*expected
        output=np.sum(output,1)
        #print np.log(output)
        return -np.sum(np.log(output))*1.0/len(predicted)
