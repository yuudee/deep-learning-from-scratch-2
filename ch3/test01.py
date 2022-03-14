import numpy as np
import sys
sys.path.append('Z:\python_practice\deep-learning-from-scratch-2-master')
sys.path.append('Z:\python_practice\deep-learning-from -scratch-2')
from common.layers import MatMul
from common.util import convert_one_hot
from common.trainer import Trainer
from ch2.util import preprocess
from util import create_contexts_target,SimpleCBOW
from common.optimizer import Adam

# 入力
c = np.array([[1,0,0,0,0,0,0]])
# 重み
W = np.random.randn(7,3)
# 中間ノード
h = np.dot(c,W)
print(h)

# 入力
c = np.array([[1,0,0,0,0,0,0]])
# 重み
W = np.random.randn(7,3)
layer = MatMul(W)
# 中間ノード
h = layer.forward(c)
print(h)

# CBOW実装
# サンプルのコンテキストデータ
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

# 重みの初期化
w_in = np.random.randn(7,3)
w_out = np.random.randn(3,7)

# レイヤの作成
in_layer0 = MatMul(w_in)
in_layer1 = MatMul(w_in)
out_layer = MatMul(w_out)

# 順伝播
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
print(s)

# コーパスからコンテキストとターゲットを作成
text = 'you say goodbye and i say hello.'
corpus,word_to_id,id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

contexts,target = create_contexts_target(corpus,window_size=1)
print(contexts)
print(target)

vocab_size = len(word_to_id)
target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)
print('target\n',target)
print('contexts\n',contexts)

# 学習コードの実装
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000
text = 'you say goodbye and i say hello'
corpus,word_to_id,id_to_word,preprocess(text)

vocab_size = len(word_to_id)
contexts,target = create_contexts_target(corpus,window_size)

target = convert_one_hot(target,vocab_size)
contexts = convert_one_hot(contexts,vocab_size)

model = SimpleCBOW(vocab_size,hidden_size)
optimizer = Adam()
trainer = Trainer(model,optimizer)

trainer.fit(contexts,target,max_epoch,batch_size)
trainer.plot()

word_vec = model.word_vecs
for word_id,word in id_to_word.items():
    print(word,word_vec[word_id])