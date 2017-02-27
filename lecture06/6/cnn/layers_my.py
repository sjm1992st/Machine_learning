from __future__ import division
import copy
import numpy as np
import numpy.random as random
import scipy.signal as sig
import cnn16.function_c as fun
def gl_uniform(size):
    return np.random.uniform(-1, 1,size=size)
class InputLayer:
    def __init__(self,input_a, num_fiters, size):
        self.input_a=input_a
        self.height = size
        self.width = size
        self.num_fiters=num_fiters
        self.output_a=self.input_a
    def backward(self, next_layer):
        #print self.output_a.shape
        [ax, ay, az, am]=self.output_a.shape
        up_weig=np.zeros([ax, next_layer.num_fiters, next_layer.size, next_layer.size])
        #print self.output_a.shape
        for i in range(ax):
            for j in range(next_layer.num_fiters):
                r12_90=np.rot90(next_layer.grad[j,:,:])
                r12_180=np.rot90(r12_90)
                up_weig[i,j,]=sig.convolve2d(self.output_a[i,0,:,:],r12_180, mode="valid")
        weig_sd=np.sum(up_weig,0)*1.0/ax
        #print weig_sd.shape
        new_weigh=np.reshape(weig_sd,(next_layer.num_fiters, next_layer.size*next_layer.size))
        next_layer.weight=next_layer.weight+ 0.1*new_weigh

class ConvolutionLayer:
    def __init__(self,input_a,num_fiters,size):
        self.input_a=input_a
        self.num_fiters=num_fiters
        self.size=size
        self.output_a=input_a
        #self.output_shape=input_shape
    def connect(self, prev_layer):
        self.stride_length=1
        self.height = ((prev_layer.height - self.size) // self.stride_length) + 1
        self.width  = ((prev_layer.width  - self.size) // self.stride_length) + 1
        self.grad=np.zeros([self.num_fiters,self.width, self.height])
        #sz=(self.num_fiters,self.size,self.size)
        #self.weight=gl_uniform(sz)
        ########c1,c2
        if prev_layer.num_fiters==self.num_fiters:
            sz=(prev_layer.num_fiters, self.size*self.size)
            self.weig=gl_uniform(sz)
            self.weight=np.zeros([self.num_fiters, prev_layer.num_fiters*self.size*self.size])
            for i in range(self.num_fiters):
                self.weight[:,i*self.size*self.size:(i+1)*self.size*self.size]=self.weig[:,0:self.size*self.size]
        else:
            sz=(self.num_fiters, prev_layer.num_fiters*self.size*self.size)
            self.weight=gl_uniform(sz)
        self.b=np.zeros([self.num_fiters,1])
    @staticmethod
    def w_dim(self,cout, weight):
        w_s=np.zeros([cout,self.num_fiters, self.size, self.size])
        for i in range(cout):
            w_s[i,]=copy.deepcopy(weight[:])
        return w_s


    def forward(self, prev_layer):
        #print
        [ax, am, ay,az]=self.input_a.shape

        #self.output=np.zeros([ax,self.num_fiters,self.width,self.height])
        """one
        for i in range(ax):
        #copy_weight=ConvolutionLayer.w_dim(self,ax, self.weight)
                for n in range(self.num_fiters):
                    if ay==self.num_fiters:
                        j=n
                    else:
                        j=0
                    self.output[i,n,:,:]=sig.convolve2d(self.input[i,j,:,:],self.weight[n,:,:], mode='valid')
        """


        #for n in range(self.num_fiters):
        self.output_a=np.zeros([ax,self.num_fiters,self.width*self.height])
        for g in range(ax):
            feature=np.zeros([prev_layer.num_fiters*self.size*self.size, self.width*self.height])
            for kl in range(prev_layer.num_fiters):
                n=0
                for j in range(ay-self.size+1):
                    for k in range(az-self.size+1):
                        pad=self.input_a[g, kl,j:j+self.size,k:k+self.size]
                        pad1=np.reshape(pad,(self.size*self.size,1))
                        feature[kl*(self.size*self.size):(kl+1)*self.size*self.size, n]=pad1[:,0]
                        n=n+1
            #print feature
            self.output_a[g,]=np.dot(self.weight, feature)
        #print self.output_a.shape
        #print self.output_a

 ##mean_up_grad[i:i+next_layer.size,j:j+next_layer.size]=up_grad[i:i+next_layer.size,j:j+next_layer.size]*next_layer.gradient[i,j]
    def backward(self, next_layer):
        self.grad=np.zeros([next_layer.up_grad.shape[0],self.num_fiters,self.width, self.height])
        #upsamping
        #print next_layer.grad.shape
        for k in range(next_layer.up_grad.shape[0]):
            for n in range(self.num_fiters):
                for i in range(0, next_layer.width):
                    for j in range(0, next_layer.height):
                        self.grad[k,n, i*next_layer.size:(i+1)*next_layer.size,j*next_layer.size:(j+1)*next_layer.size]\
                            =next_layer.up_grad[k,n,i*next_layer.size:(i+1)*next_layer.size,j*next_layer.size:(j+1)*next_layer.size]*next_layer.grad[n,i,j]
        self.grad=np.sum(self.grad,0)*1.0/next_layer.up_grad.shape[0]
        temp=np.zeros([next_layer.up_grad.shape[0],self.num_fiters, next_layer.width*next_layer.height, (self.width/next_layer.width)**2])
        #tempw=np.reshape(next_layer.grad, (next_layer.num_fiters, next_layer.width*next_layer.height))
        out_a=np.reshape(self.output_a, (next_layer.up_grad.shape[0],self.num_fiters, self.width, self.height))
        for m in range(next_layer.up_grad.shape[0]):
            for k in range(self.num_fiters):
                kp=0
                for i in range(0, self.width, next_layer.width):
                    for j in range(0, self.height, next_layer.height):
                        #print out_a.shape
                        tp=out_a[m, k, i:i+next_layer.width ,j:j+next_layer.height]
                        temp_xz=np.reshape(tp, (next_layer.width*next_layer.height, 1))
                        temp[m, k,:,kp]=temp_xz[:,0]
                        kp=kp+1
        #print  next_layer.grad.shape
        sumz=np.zeros([next_layer.up_grad.shape[0],self.num_fiters, (self.width/next_layer.width)**2])
        for m in range(next_layer.up_grad.shape[0]):
            for k in range(self.num_fiters):
                tmp_grad=np.reshape(next_layer.grad[k,], (1, next_layer.width*next_layer.height))
                sumz[m,k,]=np.dot(tmp_grad, temp[m,k,])

        zsum=np.sum(np.sum(sumz, 0),1)*1.0/((self.width/next_layer.width)**2) /next_layer.up_grad.shape[0]
        next_layer.weight=next_layer.weight + 0.1*zsum
        next_layer.b=np.sum(sumz, 1)



        #print self.input_a.shape
        #outy=np.reshape(self.output_a,(self.num_fiters, self.width, self.height))



"""
class ConvolutionLayer_C3:
    def __init__(self,input_a,num,size):
        self.input=input_a
        self.num_=num
        self.size=size
        #self.output_shape=input_shape
    def connect(self, prev_layer):
        self.stride_length=1
        self.height = ((prev_layer.height - self.size) // self.stride_length) + 1
        self.width  = ((prev_layer.width  - self.size) // self.stride_length) + 1
        sz=(self.size*self.size*prev_layer.num_fiters,self.num_)
        self.weight=gl_uniform(sz)
        self.b=np.zeros([self.num_,1])
    def forward(self, prev_layer):
        [ax,ay,az,am]=self.input.shape
        self.output=np.zeros([ax,self.num_,self.width,self.height])
        for i in range(ax):
            for n in range(self.num_):
                fiters=np.reshape([self.weight[:,n]],(prev_layer.num_fiters,self.size,self.size))
                sum_n=0
                for j in range(ay):
                    sum_n=sum_n+sig.convolve2d(self.input[i,j,:,:],fiters[j,:,:], mode='valid')
                self.output[i,n,:,:]=sigmoid(sum_n+self.b[n,0])
                    #self.output[i,n,:,:]=np.sum(self.input[i,:,:,:]+self.b[n,0])
"""

class MaxPoolingLayer():
    def __init__(self,input_a,num_fiters,size):
        self.input_a=input_a
        self.num_fiters=num_fiters
        self.size=size
        self.output_a=input_a
    def connect(self, prev_layer):
        self.num=prev_layer.num_fiters
        self.stride_length=self.size
        self.height = (prev_layer.height) // self.size
        self.width  = (prev_layer.width) // self.size
        sz=(self.num,1)
        self.weight=gl_uniform(sz)
        self.b=np.zeros([self.num,1])
        self.grad=np.zeros([self.num_fiters,self.width,self.height])

    """
    def forward(self, imgs):
        [ax,ay,az,am]=self.input.shape
        self.output=np.zeros([ax,self.num,self.width,self.height])
        for i in range(ax):
            for n in range(self.num):
                for j in range(0,ay-self.size+1,self.stride_length):
                    for k in range(0,az-self.size+1,self.stride_length):
                        self.output[i,n,j,k]=np.amx(self.input[i,n,j:j+self.size,k:k+self.size])
                self.output[i,n,j,k]=sigmoid(self.output[i,n,j,k]*self.weight[n,]+self.b[n,0])
    """
    def forward(self, prev_layer):
        #print self.input_a.shape
        [ax,ay,az]=self.input_a.shape
        self.output_a=np.zeros([ax, self.num,self.width,self.height])
        self.up_grad=np.zeros([ax, prev_layer.num_fiters,prev_layer.width,prev_layer.height])
        feature_map=np.reshape(self.input_a, (ax, prev_layer.num_fiters, prev_layer.width, prev_layer.height))
        for g in range(ax):
            for n in range(self.num):
                for j in range(0, self.width, self.stride_length):
                    for k in range(0, self.height, self.stride_length):
                        parse=feature_map[g, n, j:j+self.size, k:k+self.size]
                        self.output_a[g,n,j,k]=np.max(parse)
                        wz=np.unravel_index(np.argmax(parse),(self.size,self.size))
                        pp=np.zeros([self.size,self.size])
                        pp[wz[0], wz[1]]=1
                        self.up_grad[g, n, j:j+self.size, k:k+self.size]=pp
        #print self.output_a.shape


    def backward(self, next_layer):
        self.grad=np.zeros([self.num_fiters,self.width,self.height])
        if next_layer.num_fiters==self.num_fiters:
            new_weig=np.reshape(next_layer.weight,(next_layer.num_fiters, next_layer.size, next_layer.size))
        else:
            #diff_f_2=                                   #activation_bp[bk].diff_de()
            new_weig=np.zeros([next_layer.num_fiters,next_layer.size, next_layer.size])
            for i in range(next_layer.num_fiters):
                ds=np.reshape(next_layer.weight[i,],(self.num_fiters, next_layer.size*next_layer.size))
                new_weig[i,]=np.reshape([np.sum(ds,0)],(1,next_layer.size, next_layer.size))
            #outy=np.dot(next_layer.weight.T, next_layer.grad)
            #self.grad=self.output_a*np.reshape(outy, (self.num_fiters, self.width, self.height))
        #print  next_layer.grad.shape
        #for m in range(next_layer.grad.shape[0]):
        for i in range(self.num):
            for j in range(next_layer.num_fiters):
                r_90=np.rot90(new_weig[j,:,:])
                r_180=np.rot90(r_90)
                #print next_layer.grad.shape
                self.grad[i,:,:]=self.grad[i,:,:]+sig.convolve2d(next_layer.grad[j,:,:],r_180,mode="full")
        #self.grad[i,:,:]=self.grad[i,:,:]*1.0/next_layer.grad.shape[0]
        grad_wei=np.zeros([next_layer.num_fiters, self.num_fiters*next_layer.size*next_layer.size])
        #print self.output_a.shape
        grad_sum=0
        for k in range(self.output_a.shape[0]):
            for j in range(next_layer.num_fiters):
                grad_weigh=np.zeros([self.num_fiters, next_layer.size, next_layer.size])
                for i in range(self.num_fiters):
                    r2_90=np.rot90(next_layer.grad[j,:,:])
                    r2_180=np.rot90(r2_90)
                    grad_weigh[i,]=sig.convolve2d(self.output_a[k, i,:,:],r2_180, mode="valid")
                grad_wei[j,]=np.reshape(grad_weigh,(1,self.num_fiters*next_layer.size*next_layer.size))
            grad_sum=grad_sum+grad_wei
        next_layer.weight=next_layer.weight+0.1*grad_sum*1.0 /self.output_a.shape[0]









"""
class SquaredLossLayer(Layer):
    def __init__(self,input_shape):
    def forward(self, imgs):
    def backward(self, imgs, parents_gradient):
    def loss(self, predicted, expected):
class SoftmaxLossLayer(Layer):
    def __init__(self,input_shape):
    def forward(self, imgs):
    def backward(self, imgs, parents_gradient):
    def loss(self, predicted, expected):
"""