from __future__ import division
import numpy as np
import bp_layer.bp_top
import bp_layer.bp_input_layer as inp
import bp_layer.bp_hidden_layer as hidd
import bp_layer.bp_output_layer as out
import bp_layer.processing_data as proce
import math
import copy
class network:
    def __init__(self, node, bais, nn, cout,  mm_xtrain, mm_ytrain, mm_xval, mm_yval):
        self.node=node
        self.bais=bais
        self.nn=nn
        self.cout=cout
        self.mm_xtrain=mm_xtrain
        self.mm_ytrain=mm_ytrain
        self.mm_xval=mm_xval
        self.mm_yval=mm_yval

    @staticmethod
    def gradient_check(self, layer_copy,layer_copy2, layer_bp, data,l_data, bk):
        n_len=len(self.node)
        #self.forward(self, layer_copy, data)
        k_r=self.node[bk-1]+1
        k_c=self.node[bk]
        check_diff=np.zeros([k_r,k_c])
        for i in range(k_r):
            for j in range(k_c):
                layer_copy[bk-1].back_weight[i,j]=layer_copy[bk-1].back_weight[i,j]+10**-7
                layer_copy2[bk-1].back_weight[i,j]=layer_copy2[bk-1].back_weight[i,j]-10**-7
                self.forward(self, layer_copy, data)
                self.forward(self, layer_copy2, data)
                err2=self.l_err(layer_copy[n_len-1].output_a, layer_bp[n_len-1].output_a)
                #err=self.l_err(layer_copy2[n_len-1].classifier(),l_data)
                err=self.l_err(layer_copy2[n_len-1].output_a, layer_bp[n_len-1].output_a)

                #print  err2-err

                check_diff[i,j]=(err2-err)*1.0/(2*10**-7)
        return check_diff

    @staticmethod
    def l_err(y,y_r):
        l_e=0.5*(y_r-y)**2
        return  np.sum(l_e)
    @staticmethod
    def forward(self, layer_a, data):
        n_len=len(self.node)
        for jk in range(n_len-1):
            if jk==0:
                input_n=np.array([data])
                layer_a[0].input_a=input_n
                layer_a[0].output_a=network.c_combine(input_n)
            else:
                layer_a[jk].input_a=layer_a[jk-1].next_in()
                layer_a[jk].output_a=layer_a[jk].activation()
                layer_a[jk].output_a=network.c_combine(layer_a[jk].output_a)
        layer_a[n_len-1].input_a=layer_a[jk].next_in()
        layer_a[n_len-1].output_a=layer_a[n_len-1].activation()
    @staticmethod
    def c_combine(put_a):
        [a_x, a_y]=put_a.shape
        a_one=np.ones([a_x, 1])
        put_a=np.c_[put_a, a_one]
        return put_a
    @staticmethod
    def weight_init(a, b, bais):
        w_a=2*np.random.rand(a, b)-1
        w_a2=np.ones([1,b])*bais
        w_a=np.r_[w_a,w_a2]
        return w_a
    @staticmethod
    def mytest(self, layer_bp, n_len, mm_xval, mm_yval):
        [xt_x, xt_y]=mm_xval.shape
        m_precision=0
        lra=0
        for k in range(xt_x):
            network.forward(self, layer_bp,mm_xval[k,:])
            lra=lra+self.l_err(layer_bp[n_len-1].softmax(), mm_yval[k, :])
            sub=np.sum(abs(layer_bp[n_len-1].classifier()-mm_yval[k, :]))
            if sub==0:
                m_precision=m_precision+1
        return  lra*1.0/xt_x,m_precision*1.0/xt_x


    def net(self):   ##(a,b,c)
        n_len=len(self.node)
        [xt_x, xt_y]=self.mm_xtrain.shape
        input_arra=np.zeros([1,xt_y])
        layer_bp=[]
        pb_mse=[]
        md_precision=[]
        for i in range(n_len-1):
            if i==0:
                a_in=inp.input_layer(input_arra, input_arra, network.weight_init(self.node[0], self.node[1], self.bais))
                layer_bp.append(a_in)
            else:
                a_hidden=hidd.hidden_layer(input_arra, input_arra, network.weight_init(self.node[i], self.node[i+1], self.bais))
                layer_bp.append(a_hidden)
        a_out=out.output_layer(input_arra,input_arra)
        layer_bp.append(a_out)
        for pi in range(self.cout):
            for k in range(xt_x):
                network.forward(self, layer_bp,self.mm_xtrain[k,:])
                layer_copy=copy.deepcopy(layer_bp)
                layer_copy2=copy.deepcopy(layer_bp)
                for bk in range(n_len-1,0,-1):
                    if bk==n_len-1:
                        yr=layer_bp[n_len-1].diff_out_1(layer_bp[n_len-1].classifier(), self.mm_ytrain[k, :])        ##(1,10) ##(1,23)
                        diff_o=np.dot(yr.T,layer_bp[bk-1].output_a)
                        if bk==1:
                            check_diff=network.gradient_check(self, layer_copy,layer_copy2,layer_bp, self.mm_xtrain[k,:],self.mm_ytrain[k, :], bk)
                            #print diff_o.T
                            #print check_diff
                            print diff_o.T-check_diff
                        layer_bp[bk-1].back_weight=layer_bp[bk-1].back_weight+self.nn*diff_o.T                               ##(23,10)
                    else:
                        yr=layer_bp[bk].diff_out_1(yr)
                        diff_h=np.dot(yr.T,layer_bp[bk-1].output_a)
                        if bk==2:
                            check_diff=network.gradient_check(self, layer_copy,layer_copy2, layer_bp,self.mm_xtrain[k,:],self.mm_ytrain[k, :], bk)
                            #print diff_h.T
                            #print check_diff
                            print diff_h.T-check_diff
                        layer_bp[bk-1].back_weight=layer_bp[bk-1].back_weight+self.nn*diff_h.T

            lra, mpe=network.mytest(self, layer_bp, n_len, self.mm_xval, self.mm_yval)
            print lra
            print mpe

            pb_mse.append(lra)
            md_precision.append(mpe)
        return pb_mse, md_precision
