import numpy as np
import sys
sys.path.append('Z:\python_practice\deep-learning-from-scratch-2-master')
from util import UnigramSampler
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from util import CBOW
from dataset import ptb
from common.util import create_contexts_target,most_similar,analogy




# 特定の行を抜き出す
W = np.arange(21).reshape(7,3)
print(W)
print(W[2])

idx = np.array([1,0,3,0])
print(W[idx])

# 確率分布に従ったサンプリング
num1 = np.random.choice(10)
print(num1)

words = ['you','say','goodbye','i','hello','.']
word1 = np.random.choice(words)
print(word1)

# 重複アリで5つ
word2 = np.random.choice(words,size=5)
print(word2)

# 重複なしで5つ
word3 = np.random.choice(words,size=5,replace=False)
print(word3)

# 確率分布に従って
p = [0.5,0.1,0.05,0.2,0.05,0.1]
word4 = np.random.choice(words,p=p)
print(word4)

# 0.75乗
p = [0.7,0.29,0.01]
pow_p = np.power(p,0.75)
new_p = pow_p / np.sum(pow_p)
print(new_p)

# UnigramSampler試し
corpus = np.array([0,1,2,3,4,1,2,3])
power = 0.75
sample_size = 2

sampler = UnigramSampler(corpus,power,sample_size)
target = np.array([1,3,0])
negative_sample = sampler.get_negative_sample(target)
print(negative_sample)

# # CBOW学習
# window_size = 5
# hidden_size = 100
# batch_size = 100
# max_epoch = 10

# # データの読み込み
# corpus,word_to_id,id_to_word = ptb.load_data('train')
# vocab_size = len(word_to_id)

# contexts,target = create_contexts_target(corpus,window_size)

# # モデル生成
# model = CBOW(vocab_size,hidden_size,window_size,corpus)
# optimizer = Adam()
# trainer = Trainer(model,optimizer)

# trainer.fit(contexts,target,max_epoch,batch_size)
# trainer.plot()

# # 分散表現の取得
# word_vec = model.word_vec

# params = {}
# params['word_vecs'] = word_vec.astype(np.float16)
# params['word_to_id'] = word_to_id
# params['id_to_word'] = id_to_word
# pkl_file = 'cbow_params.pkl'
# with open(pkl_file,'wb') as f:
#     pickle.dump(params,f,-1)

pkl_file = 'cbow_params.pkl'

with open(pkl_file,'rb') as f:
    params = pickle.load(f)
    word_vec = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']
    
querys = ['you','year','car','toyota']
for query in querys:
    most_similar(query,word_to_id,id_to_word,word_vec,top=5)
    
print(analogy('king','man','queen',word_to_id,id_to_word,word_vec))
print(analogy('take','took','go',word_to_id,id_to_word,word_vec)) 
