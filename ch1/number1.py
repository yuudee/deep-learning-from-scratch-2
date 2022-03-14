import numpy as np


def softmax(x):  # ソフトマックス
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)  # オーバーフロー対策
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):  # 交差エントロピー誤差
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    # t=[2,5,7]の時 [0, 2], [1, 5], [2, 7] のデータが抽出される。batch_size=3より。
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:  # 全結合層(アフィンレイヤ)
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W)+b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

"""
class Twolayernet:  # ニューラルネットワークの生成
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        w1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        w2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイヤの生成
        self.layer = [Affine(w1, b1), sigmoid(), Affine(w2, b2)]
        self.params = []
        for layer in self.layer:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layer:
            x = layer.forward(x)
        return x
"""

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmaxの出力
        self.t = None  # 教師ラベル

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class SGD:
    '''
    確率的勾配降下法（Stochastic Gradient Descent）
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


"""
# main
x = np.random.randn(10, 2)
model = Twolayernet(2, 4, 3)
s = model.predict(x)
print(s)
"""