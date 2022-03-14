import numpy as np
import sys
sys.path.append("Z:\python_practice\deep-learning-from-scratch-2-master")
from common.layers import Embedding,Softmax

class RNN:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np,np.zeros_like(b)]
        self.cache = None
    
    def forward(self,x,h_prev):
        Wx,Wh,b = self.params
        t = np.dot(h_prev,Wh) + np.dot(x,Wx) + b
        h_next = np.tanh(t)
        
        self.cache = (x,h_prev,h_next)
        return h_next
    
    def backward(self,dh_next):
        Wx,Wh,b = self.params
        x,h_prev,h_next = self.cache
        
        dt = dh_next * (1 - h_next**2)
        db = np.sum(dt,axis=0)
        dWh = np.dot(h_prev.T,dt)
        dh_prev = np.dot(dt,Wh.T)
        dWx = np.dot(x.t,dt)
        dx = np.dot(dt,Wx.T)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
class TimeRNN:
    def __init__(self,Wx,Wh,b,stateful=False):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.layers = None
        
        self.h,self.dh = None,None
        self.stateful = stateful
    
    def set_state(self,h):
        self.h = h
        
    def reset_state(self):
        self.h = None
        
    def forward(self,xs):
        Wx,Wh,b = self.params
        N,T,D = xs.shape    #バッチサイズN、時系列の個数T、入力ベクトルの次元数D
        D,H = Wx.shape
        
        self.layers = []
        hs = np.empty((N,T,H),dtype='f')
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H),dtype='f')
            
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:],self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)
        return hs
    
    def backward(self,dhs):
        Wx,Wh,b = self.params
        N,T,H = dhs.shape
        D,H = Wx.shape
        
        dxs = np.empty((N,T,D),dtype='f')
        dh = 0
        grads = [0,0,0]
        for t in reversed(range(T)):    #上位のRNNレイヤから
            layer = self.layers[T]
            dx,dh = layer.backward(dhs[:,t:]+dh)
            dxs[:,t,:] = dx
            
            for i,grad in enumerate(layer.grads):
                grads[i] += grad
                
        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        
        return dxs
    
class TimeEmbedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W
        
    def forward(self,xs):
        N,T = xs.shape  #バッチサイズN、時系列の個数T
        V,D = self.W.shape
        
        out = np.empty((N,T,D),dtype='f')
        self.layers = []
        
        for t in  range(T):
            # Embedレイヤの生成
            layer = Embedding(self.W)
            
            # 順伝播の計算
            out[:,t,:] = layer.forward(xs[:,t])
            
            # t番目のレイヤの格納
            self.layers.append(layer)
        return out
    
    def backward(self,dout):
        N,T,D = dout.shape
        
        dW = np.zeros_like(self.W)
        for t in range(T):
            layer = self.layers[t]
            
            # 逆伝播の計算
            layer.backward(dout[:,t,:])
            
            dW += layer.grads[0]
        self.grads[0][...] = dW
        return None            
    
class TimeAffine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
    
    def forward(self,x):
        N,T,D = x.shape #バッチサイズN、時系列の個数T、入力ベクトルの次元数D
        W,b = self.params 
        
        rx = x.reshape(N*T,-1)
        out = np.dot(rx,W) + b
        
        self.x = x
        return out.reshape(N,T,-1)
    
    def backward(self,dout):
        x = self.x
        N,T,D = x.shape
        W,b = self.params
        
        dout = dout.reshape(N*T,-1)
        rx = x.reshape(N*T,-1)
        
        db = np.sum(dout,axis=0)
        dW = np.dot(rx.T,dout)
        dx = np.dot(dout,W.T)
        dx = dx.reshape(*x.shape)
        
        self.grads[0][...] = dW
        self.grads[0][...] = db
        
        return dx
    
class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.cache = []
        self.ignore_label = -1
    
    def forward(self,xs,ts):
        N,T,V = xs.shape
        
        if ts.ndim == 3:
            ts = ts.argmax(axis=2)
            
        mask = (ts != self.ignore_label)
        
        xs = xs.reshape(N*T,V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)
        
        ys = Softmax(xs)
        ls = np.log(ys[np.arange(N*T),ts])  #正解ラベルが1の部分についてのみ対数をとる
        ls*= mask
        loss = -np.sum(ls)
        loss /= mask.sum()
        
        self.cache = (ts,ys,mask,(N,T,V))
        return loss
    
    def backward(self,dout=1):
        ts,ys,mask,(N,T,V) = self.cache
        
        dx  =ys
        dx[np.arange(N*T),ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:,np.newaxis]
        
        dx = dx.reshape(N,T,V)
        
        return dx