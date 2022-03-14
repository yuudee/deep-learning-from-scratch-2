import sys
sys.path.append('Z:\python_practice\deep-learning-from-scratch-2-master')
from number1 import SGD
from  Trainer import Trainer
from dataset import spiral
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x,t =spiral.load_data()
model = TwoLayerNet(input_size=2,hidden_size=hidden_size,output_size=3)
optimazer = SGD(lr=lr)

trainer = Trainer(model,optimazer)
trainer.fit(x,t,max_epoch,batch_size,eval_interval=20)
trainer.plot()