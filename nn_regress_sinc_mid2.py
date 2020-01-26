# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NN3Layer(chainer.Chain):
    def __init__(self):
        w = I.Normal(scale=0.1)
        super(NN3Layer, self).__init__(
                l1 = L.Linear(2, 100, initialW=w),
                l2 = L.Linear(100, 100, initialW=w),
                l3 = L.Linear(None, 1, initialW=w),
        )
        
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y

    
def plot_data_3d(XX,YY,ZZ):

    fig = plt.figure()
    ax = Axes3D(fig)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    
    ax.plot_wireframe(XX, YY, ZZ, color="blue",linewidth=0.3)
    plt.show()

    
def main():
    
    epoch = 250
    batchsize = 50
    
    #データの作成(3次元のsinc関数)
    N = 10
    x = np.arange(-N,N,0.2)
    y = np.arange(-N,N,0.2)
    XX, YY = np.meshgrid(x, y)
    RR = np.sqrt(XX*XX + YY*YY)
    ZZ = np.sin(RR) / RR
    
#    plot_data_3d(XX, YY, ZZ)
    
    
    #学習データ作成（作成したデータから10000点をチョイス)
    Nsmp = 10000
    Nidx = x.size
    idx = range(Nidx)
    i = random.choices(idx, k=Nsmp)
    j = random.choices(idx, k=Nsmp)
    
    trainx = np.empty((Nsmp,2), dtype=np.float32)
    for k in range(Nsmp):
        trainx[k] = np.array((XX[i[k]][j[k]], YY[i[k]][j[k]]))

    trainy = np.empty((Nsmp,1), dtype=np.float32)
    for k in range(Nsmp):
        trainy[k][0] = ZZ[i[k]][j[k]]
    
    train = chainer.datasets.TupleDataset(trainx, trainy)
    
    #テストデータ作成（作成したデータから1000点をチョイス
    Nsmp = 1000
    idx = range(50)
    i = random.choices(idx, k=Nsmp)
    j = random.choices(idx, k=Nsmp)
    
    testx = np.empty((Nsmp,2), dtype=np.float32)
    for k in range(Nsmp):
        testx[k] = np.array((XX[i[k]][j[k]], YY[i[k]][j[k]]))

    testy = np.empty((Nsmp,1), dtype=np.float32)
    for k in range(Nsmp):
        testy[k][0] = ZZ[i[k]][j[k]]

    test = chainer.datasets.TupleDataset(testx, testy)
        

    #ニューラルネットワークの登録
    model = L.Classifier(NN3Layer(), lossfun=F.mean_squared_error)  #平均2乗誤差
    model.compute_accuracy = False
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    
    #イテレータの登録
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
    
    #トレーナーの登録
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (epoch, 'epoch'))
    
    #学習状況の表示や保存
    trainer.extend(extensions.Evaluator(test_iter, model))
    #誤差のグラフ
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='loss.png'))
    #ログ
    trainer.extend(extensions.LogReport())

    #計算状態の表示
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

    #エポック毎にトレーナーの状態を保存する
    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))    
    
#    chainer.serializers.load_npz("result/snapshot_iter_40000", trainer)
    
    #学習開始
    trainer.run()
    
    #途中状態の保存
    chainer.serializers.save_npz("result/nn.model", model)
    
    #学習結果の評価
    ZZpred = np.empty((Nidx,Nidx), dtype=np.float32)

    for i in range(Nidx):
        for j in range(Nidx):
            evalx_np = np.array((XX[i][j], YY[i][j]), dtype=np.float32)
            evalx = chainer.Variable(evalx_np.reshape(1,2))
            result = model.predictor(evalx)
            ZZpred[i][j] = result.data
            
    plot_data_3d(XX,YY,ZZpred)
    
if __name__ == '__main__':
    main()
    
