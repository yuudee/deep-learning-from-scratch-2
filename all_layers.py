#順伝播、逆伝播は計算グラフで考えるとわかりやすい！
from numpy import dtype, negative
from config import *
import sys
sys.path.append('Z:\python_practice\deep-learning-from-scratch-2-master\ch04')
from negative_sampling_layer import UnigramSampler

#シグモイド関数
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

#ReLu関数
def relu(x):
    return np.maximum(0,x)

# バッチ版ソフトマックス関数
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1,keepdims=True) #オーバフロー対策のため最大値を引く、keepdimsは出力時に次元数を固定する
        x = np.exp(x)
        x /= x.sum(axis=1,keepdims=True)    #axis=1は横方向
    elif x.ndims==1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
        
# 交叉エントロピー誤差(y:出力 t:正解)
#size:行列全体の個数,shape:x行y列
def cross_entropy_error(y,t):
    # 1次元の時２次元配列に変換
    if y.ndim == 1:
        t = t.reshape(1,t.size) #1行t.size列
        y = y.reshape(1,y.size) #1行t.size列
    
    # 教師データがone-hotベクトルの場合
    if t.size == y.size:
        t = t.argmax(axis=1)    #one-hotベクトルは一つだけ1なのでそこが最大に
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size

# MatMulレイヤ
class MatMul:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self,x):
        W, = self.params
        out = np.dot(x,W)
        self.x = x
        return out
    
    def backward(self,dout):
        W, = self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        self.grads[0][...] = dW
        return dx
    
# Affineレイヤ
class Affine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
        
    def forward(self,x):
        W,b = self.params
        out = np.dot(x,W) + b
        self.x = None
    
    def backward(self,dout):
        W,b = self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        db = np.sum(dout,axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
    
# ソフトマックスレイヤ
class Softmax:#ゼロから1 P280
    def __init__(self):
        self.params,self.grads = [],[]
        self.out = None
    
    def forward(self,x):
        self.out = softmax(x)
        return self.out
    
    def backward(self,dout):
        dx = self.out * dout
        sumdx = np.sum(dx,axis=1,keepdims=True)
        dx -= self.out * sumdx
        return dx

#SoftmaxWithLossレイヤ
class SoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads = [],[]
        self.y = None   #softmax出力
        self.t = None   #教師ラベル
        
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        
        #教師ラベルがone-hotベクトルの場合正解のインデックスを取得
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        loss = cross_entropy_error(self.y,self.t)
        return loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size),self.t] -= 1   #one-hotベクトルなので正解の時だけ1引く
        dx *= dout  #sumの逆伝播？
        dx = dx / batch_size
        
        return dx
 
# Sigmoidレイヤ   
class Sigmoid:  #ゼロから1 P146
    def __init__(self):
        self.params,self.grads = [],[]
        self.out = None
        
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout * (1 - self.out) * self.out
        return dx

#SigmoidWithLossレイヤ  
class SigmoidWithLoss:
    def __init__(self):
        self.params,self.grads = [],[]
        self.loss = None
        self.y = None   #sigmoidの出力
        self.t = None   #教師データ
    
    def forward(self,x,t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        
        self.loss = cross_entropy_error(np.c_[1 - self.y,self.y],self.t)    #横方向に結合,損失関数の式から
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        
        dx = (self.y - self.t) * dout / batch_size
        return dx
    
#Embeddingレイヤ
class Embedding:    #idxに該当する行を抜き出す(分散表現の取得)
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self,idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self,dout):
        dW, = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW,self.idx,dout)
        else:
            np.add.at(dW,self.idx,dout)
        # for i,word_id in enumerate(self.idx):
        #     dW[word_id] += dout[i]と同じ
        return None
    
#EmbeddingDotレイヤ
class EmbediingDot:
    def __init__(self,W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self,h,idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h,idx,axis=1)
        
        self.cache = (h,target_W)
        return out
    
    def backward(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
    
#NegativeSamplingWithLossレイヤ
class NegativeSamplingLoss:
    def __init__(self,W,corpus,power=0.75,sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus,power,sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size+1)]    #2値分類のためsigmoid関数
        self.embed_dot_layers = [EmbediingDot(W) for _ in range(sample_size+1)]
        
        self.params,self.grads = [],[]
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def forward(self,h,target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.getnegative_sample(target)
        
        # 正例について
        score = self.embed_dot_layers[0].forward(h,target)
        correct_label = np.ones(batch_size,dtype=np.int32)  #正例だから1で初期化
        loss = self.loss_layers[0].forward(score,correct_label)
        
        # 負例について
        negative_label = np.zeros(batch_size,dtype=np.int32)    #負例なので0で初期化
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[i+1].forward(h,negative_target)
            loss += self.loss_layers[i+1].forward(score,negative_label)
            
        return loss
    
    def backward(self,dout=1):
        dh = 0
        for l0,l1 in zip(self.loss_layers,self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        
        return dh
    
#CBOWレイヤ
class CBOW:
    def __init__(self,vocab_size,hidden_size,window_size,corpus):
        V,H = vocab_size,hidden_size
        
        # 重みの初期化
        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.randn(V,H).astype('f')
        
        # レイヤの生成
        self.in_layers = []
        for i in range(2*window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out,corpus,power=0.75,sample_size=5)
        
        # 重みと勾配をまとめる
        layers = self.in_layers + [self.ns_loss]
        self.params,self.grads = [],[]
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # メンバ変数に単語の分散表現
        self.word_vecs = W_in
        
    def forward(self,contexts,target):
        h = 0
        for i,layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        #https://www.anarchive-beta.com/entry/2020/09/14/190000#32-%E3%82%B7%E3%83%B3%E3%83%97%E3%83%AB%E3%81%AAword2vec
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h,target)
        return loss
            
    def backward(self,dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None