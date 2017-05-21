# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:02:00 2017

@author: LSJ
"""
import random
import csv
import time
import math


class GA(object):
    '''
    遗传算法
    '''
    
    def __init__(self,lenofdna,popsize,pofcross,pofmutate):
        self.popsize=popsize
        self.pofcross=pofcross
        self.pofmutate=pofmutate
        self.lenofdna=lenofdna
        self.fit=list(range(popsize)) #value of fitness
        self.pop=[''.join([str(random.randint(0,1)) for i in range(lenofdna)]) \
        for j in range(popsize)] #type(pop)=list, number=popsize ,dna is binary 
        self.newpop = list(range(popsize))
        
            
    def select(self):
        optimum=self.fit.index(max(self.fit))
        self.newpop[0] = self.pop[optimum]
        accu=list(range(self.popsize))
        diffit=[i-min(self.fit) for i in self.fit]
        er=max(diffit)/len(diffit)/2
        accu[0]=diffit[0]+er
        for i in range(1,self.popsize):
            accu[i]=accu[i-1]+diffit[i] 
        accu=[i/max(accu)+0.0001 for i in accu]
        for i in range(1,len(self.newpop)):
            temp = random.random()
            indexpop=[i>temp for i in accu].index(True)
            self.newpop[i] = self.pop[indexpop]
        self.pop=self.newpop.copy()
        
    def crossover(self):
        for i in range(1,self.popsize):
            if random.random() < self.pofcross:
                cpoint=random.randint(1,self.lenofdna-2)
                matchi=random.randint(0,self.popsize-1)
                self.pop[i],self.pop[matchi]=\
                self.pop[i][:cpoint]+self.pop[matchi][cpoint:],self.pop[matchi][:cpoint]+self.pop[i][cpoint:]
    
    def mutation(self):
        for i in range(1,self.popsize):
            self.pop[i]=''.join([(j,('0','1')[j=='0'])[random.random()<self.pofmutate] for j in self.pop[i]])
    
    def bestone(self):
        optimum=self.fit.index(max(self.fit))
        return self.pop[optimum]
    
    def averagefit(self):
        return sum(self.fit)/self.popsize
###############################################自定义适应度函数，基因解码函数#################
def dnadecode(pop):
    hn=[]
    maxdna=2**10-1
    for dna in pop:
        p1=int(dna[:10],2)/maxdna
        p1=p1*3-1


        hn.append(p1)


    return hn

def calfitness(x):
    y=x*math.sin(math.pi*10*x)+2
    return y


if __name__ == '__main__':
###############################################遗传算法流程 #################
    ga=GA(10,50,0.6,0.01)
    starttime=time.time()
    epoch=0
    psize=ga.popsize
    historyfit=[]
    while epoch < 500 : #终止条件
        hn=dnadecode(ga.pop) #解码
        ga.fit=list(map(calfitness,hn)) # 计算适应度函数
        ga.select() #选择
        ga.crossover()  #交叉
        ga.mutation()   #变异
        epoch+=1        #代数     
        historyfit.append( ( ga.averagefit() , max(ga.fit) ) )
    print([hn[ga.fit.index(max(ga.fit))]])
###############################################################################
    with open('history.csv','w',newline='') as cf:
        wcf=csv.writer(cf)
        wcf.writerow( ['epoch','averagefit','bestone'])
        for i,j in enumerate(historyfit):
            wcf.writerow((i,j[0],j[1]))