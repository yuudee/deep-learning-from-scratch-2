import numpy as np
import sys
sys.path.append('Z:\python_practice\deep-learning-from-scratch-2-master')
from common.layers import MatMul,SoftmaxWithLoss

# コンテキストとターゲットを作成する関数
def create_contexts_target(corpus,window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size,len(corpus)-window_size):
        cs = []
        for t in range(-window_size,window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)

class SimpleCBOW:
    def __init__(self,vocab_size,hidden_size):
        V,H = vocab_size,hidden_size
        
        #重みの初期化
        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.randn(H,V).astype('f')
        
        # レイヤの生成
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        
        # 全ての重みと勾配をリストにまとめる
        layers = [self.in_layer0,self.in_layer1,self.out_layer]
        self.params,self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # メンバ変数に単語の分散表現を設定
        self.word_vecs = W_in
        
    def forward(self,contexts,target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = 0.5 * (h0 + h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score,target)
        return loss
    
    def backward(self,dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None