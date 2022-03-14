from util import *
import sys
sys.path.append('Z:\python_practice\deep-learning-from-scratch-2-master')
from dataset import ptb

#コーパスの前処理
text = 'You say goodbye and I say Hello.'

corpus,word_to_id,id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

#共起行列の生成
C = np.array([
    [0,1,0,0,0,0,0],
    [1,0,1,0,1,1,0],
    [0,1,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,1,0,0,0],
    [0,1,0,0,0,0,1],
    [0,0,0,0,0,1,0]
],dtype=np.int32)

print(C[0])
print(C[4])
print(C[word_to_id['goodbye']])

#共起行列の作成
co_matrix = create_co_matrix(corpus,len(word_to_id),window_size=1)
print(co_matrix)

#cos類似度
C0 = C[word_to_id['you']]
C1 = C[word_to_id['i']]
print(cos_similarity(C0,C1))

#類似度上位5つ
text = 'You say goodbye and I say Hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)
most_similar('you',word_to_id,id_to_word,C)

#共起行列からppmiへの変換
text = 'You say goodbye and I say Hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

#SVDによる次元削減
text = 'You say goodbye and I say Hello.'
corpus,word_to_id,id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus,vocab_size)
W = ppmi(C)
U,S,V = np.linalg.svd(W)

print(C[0])
print(W[0])
print(U[0])
print(U[0,1:])

for word ,word_id in word_to_id.items():
    plt.annotate(word,(U[word_id,0],U[word_id,1]))
plt.scatter(U[:,0],U[:,1],alpha=0.5)
#plt.show()

#PTBデータセット
corpus,word_to_id,id_to_word = ptb.load_data('train')
print('corpus_size:',len(corpus))
print('corpus[:30]',corpus[:30])
print()
print('id_to_word[0]:',id_to_word[0])
print('id_to_word[1]:',id_to_word[1])
print('id_to_word[2]:',id_to_word[2])
print()
print("word_to_id['car']",word_to_id['car'])
print("word_to_id['happy']",word_to_id['happy'])
print("word_to_id['lexus']",word_to_id['lexus'])

# PTBデータセットでの評価
window_size = 2
wordvec_size = 100

corpus,word_to_id,id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence...')
C = create_co_matrix(corpus,vocab_size,window_size)
print('calculating PPMI...')
W = ppmi(C,verbose=True)
print('calculating SVD...')
try:
    from sklearn.utils.extmath import randomized_svd
    U,S,V = randomized_svd(W,n_components=wordvec_size,n_iter=5,random_state=None)
except ImportError:
    U,S,V = np.linalg.svd(W)
    
word_vecs = U[:,:wordvec_size]
querys = ['you','year','car','toyota']
for query in querys:
    most_similar(query,word_to_id,id_to_word,word_vecs,top=5)