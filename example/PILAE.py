# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:28:13 2017

@author: LSJ
"""

import scipy.io as sio    
import numpy as np 
from sklearn import preprocessing
import random
from ActivationFunc import ActivationFunc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

class PILAE(object):

    def __init__(self,HiddernNeurons,batchsize,k,para,actFun):
        self.HiddernNeurons = HiddernNeurons          # 隐层神经元
        self.batchsize = batchsize            # 批次大小
        self.k = k             # 误差
        self.para = para            # sigmoid p值
        self.actFun = actFun            # 激活函数
        self.layers = len(HiddernNeurons)   #隐层层数
        self.w=list(range(self.layers))        #权重
        self.preiS=list(range(self.layers))        #is
        self.InputWeight=list(range(self.layers))       #输入权重

    def makebatches(self, data):
        '''
        将样本分批次
        '''
        randomorder=random.sample(range(len(data)),len(data))   
        numbatches=int(len(data)/self.batchsize)
        for batch in range(numbatches):
            order=randomorder[batch*self.batchsize:(batch+1)*self.batchsize]
            yield data[order,:]

    def train(self, data):
        '''
        训练样本
        '''
        for numbatch,delta_X in enumerate(self.makebatches(data)):
            for i in range(self.layers):
                U,s,V=np.linalg.svd(delta_X)
                V=V.T
                V=V[:,:self.HiddernNeurons[i]]
                S=np.zeros((U.shape[0],V.shape[0] ))
                S[:s.shape[0],:s.shape[0]]=np.diag(s)
                S=S[:self.HiddernNeurons[i],:self.HiddernNeurons[i]]
                S[S!=0]=1/S[S!=0]
                self.InputWeight[i] = V.dot(S)
                
                
                if numbatch ==0:
                    H = delta_X.dot(self.InputWeight[i])
                    delta_H = ActivationFunc(H,self.actFun[i],self.para[i])
                    iS = np.linalg.inv( delta_H.T.dot(delta_H) + np.eye(delta_H.shape[1])*self.k[i])
                    OW = iS.dot(delta_H.T) .dot(delta_X)
                else:
                    H = delta_X.dot( self.w[i].T)
                    delta_H = ActivationFunc(H,self.actFun[i],self.para[i])
                    iC = np.linalg.inv(np.eye(delta_H.shape[0]) + delta_H.dot(self.preiS[i]).dot(delta_H.T))
                    iS = self.preiS[i] - self.preiS[i].dot(delta_H.T).dot(iC).dot(delta_H).dot(self.preiS[i])          
                    alpha =np.eye(delta_H.shape[1]) - self.preiS[i].dot(delta_H.T).dot(iC).dot(delta_H)
                    belta = self.preiS[i].dot( np.eye(delta_H.shape[1]) - delta_H.T.dot( iC).dot(delta_H).dot(self.preiS[i]) ).dot(delta_H.T).dot(delta_X)
                    OW = alpha.dot(self.w[i]) + belta
                    
                self.w[i] = OW
                self.preiS[i]=iS
                
                tempH = delta_X.dot(OW.T)
                delta_X = ActivationFunc(tempH,self.actFun[i],self.para[i])
                delta_X = preprocessing.scale(delta_X.T)
                delta_X = delta_X.T


    def feature_extrackted(self,x):
        '''
        自编码器的特征提取
        '''
        feature=x
        for i in range(self.layers):
            feature=feature.dot(self.w[i].T)
            feature=ActivationFunc(feature,self.actFun[i],self.para[i])
        return feature



if __name__ == '__main__':

    HiddernNeurons = [600,500,400]
    batchsize =1000
    k = [1e-7,1e-4,1e-3]
    para = [1.5,1.5,1.5]
    actFun = ['sin','sin','sin']
    #导入数据
    X = sio.loadmat('G:/BaiduYunDownload/PILAEMNIST/mnistX.mat')
    X=X['X']
    Label = sio.loadmat('G:/BaiduYunDownload/PILAEMNIST/mnistnumY.mat');
    Label=Label['numY']
    idx = sio.loadmat('G:/BaiduYunDownload/PILAEMNIST/mnistidx.mat');
    # 数据预处理
    train_X = X[:,idx['trainidx'][0]-1]
    train_Y = Label[:,idx['trainidx'][0]-1]
    train_Y = np.array([[int(i) for i in train_Y[0]]])
    test_X = X[:,idx['teidx'][0]-1]
    test_Y = Label[:,idx['teidx'][0]-1]
    test_Y = np.array([[int(i) for i in test_Y[0]]])
    
    P = preprocessing.scale(train_X)
    T = train_Y
    TVP = preprocessing.scale(test_X)
    TVT = test_Y
    
    label=np.array([list(set(list(T[0])))])
    
    tempT=np.zeros((len(label[0]),len(T[0])))
    for i in range(len(tempT[0])):
        tempT[int(T[0,i]),i]=1
    T=tempT*2-1
    
    tempT=np.zeros((len(label[0]),len(TVT[0])))
    for i in range(len(tempT[0])):
        tempT[int(TVT[0,i]),i]=1
    TVT=tempT*2-1
    data = P.T
    
######################   模型训练   #######################################     
    t1=time.time()
    pilae=PILAE(HiddernNeurons,batchsize,k,para,actFun) #初始化PILAE模型
    pilae.train(data)       #训练模型
    t2=time.time()
    print("time cost of training PILAE: %.4f" % (t2-t1))
    featureofdate=pilae.feature_extrackted(data)        #提取特征训练集
    testfeatureofdate=pilae.feature_extrackted(TVP.T)        #提取特征测试集
######################   特征分类   #######################################  
        
    reg = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    
    reg.fit(featureofdate, Label[0,0:60000])
    # np.savetxt('coef_softmax_sklearn.txt', reg.coef_, fmt='%.6f')  # Save coefficients to a text file
    ptrain = reg.predict(featureofdate)
    print("Accuracy of train set: %f" % accuracy_score(ptrain, Label[0,0:60000]))
    
    #reg.fit(testfeatureofdate, Label[0,60000:])
    ptest = reg.predict(testfeatureofdate)
    print("Accuracy of test set: %f" % accuracy_score(ptest, Label[0,60000:]))









