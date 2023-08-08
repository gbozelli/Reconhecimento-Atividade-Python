from matplotlib import pyplot
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def sigmoidFunction(Theta,Theta0,X):
  a =  np.dot(Theta,X) + Theta0
  Y = 1/(1+np.exp(a))
  return Y

def copy(X):
  Xis = []
  for i in X:
    Xis.append(i)
  Xis = np.transpose(Xis)
  Xs = Xis
  Xis = []
  Xis.append(Xs)
  return Xis

def dG(X,Y,K):
  X = np.transpose(X)
  N = len(Y)
  Theta = np.zeros((K,1))
  Theta0 = 0
  D = 1
  lam = 10
  while(D>1e-9):
    YPred = np.dot(X,Theta) + Theta0
    for k in range(1,K+1):
      D = lam*(2/N)*np.sum(np.matmul(np.transpose(X[k]),(Y-YPred)))
      Theta[k-1] += D
    Theta0 += lam*(2/N)*np.sum(Y-YPred)
  return Theta, Theta0

def binarizeData(Y,ref):
  Ya = []
  for i in range(len(Y)):
    if Y[i] == ref:
      Ya.append(1)
    else:
      Ya.append(0)
  return Ya

def correlationMatrix(X):
  correlations = X.corr()
  fig = pyplot.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(correlations, vmin=-1, vmax=1)
  fig.colorbar(cax)
  pyplot.show()

def trainOvA(X,Y):
  Ybin = []
  Models,Models0=[],[]
  for i in range(classes):
    Ybin.append(binarizeData(Y,i))
  for i in range(classes):
    YCurrent = Ybin[i]
    Theta, Theta0 = dG(X,YCurrent,10)
    Models.append(Theta)
    Models0.append(Theta0)
  return Models,Models0


def evaluateModel(Models1,Models0,Xtest,Ytest):
  eval, Ybin = [], []
  for i in range(classes):
    Ybin.append(binarizeData(Ytest,i))
  for i in range(classes):
    CurrentModel = Models1[i]
    CurrentModel0 = Models0[i]
    Ypred = sigmoidFunction(CurrentModel,CurrentModel0,Xtest)
    eval.append(Ypred)
  for i in range(0,classes):
    for j in range(len(Ypred)):
      if eval[i][j]>0.5:
        eval[i][j] = 1
      else:
        eval[i][j] = 0
  Y,Ypred = Ybin, eval
  for i in range(classes):
    TN,TP,FN,FP = CFMatrix(Y[i],Ypred[i])
    print("Fscore class ",i+1," = ",fScore(TN,FP,FN))
  return Ybin, eval

def CFMatrix(label,yTest):
  lenTrain = 0
  TN,TP,FN,FP = 0,0,0,0
  for i in range(len(yTest)):
    if label[i] == yTest[lenTrain+i]:
      if label[i] == 0:
        TN += 1
      if label[i] == 1:
        TP += 1
    if label[i] != yTest[lenTrain+i]:
      if label[i] == 0:
        FN += 1
      if label[i] == 1:
        FP += 1
  return TN,TP,FN,FP

def fScore(TP,FP,FN):
  accuracy, recall = TP/(TP+FP), TP/(TP+FN)
  fScore = (2*accuracy*recall)/(accuracy+recall)
  return fScore

features = 10
classes = 6

Xtrain = pd.read_csv('X_train.csv',sep=' ',header=None)
Ytrain = np.array(pd.read_csv('y_train.csv',sep=' ',header=None))
Xtest = pd.read_csv('X_test.csv',sep=',',header=None)
Ytest = np.array(pd.read_csv('y_test.csv',sep=' ',header=None))

Xtrain, Xtest = np.transpose(Xtrain)[1:features+1], np.transpose(Xtest)[1:features+1]

Models,Models0 = trainOvA(Xtrain,Ytrain)
Models = np.array(Models)
Models = Models.reshape(classes,10*1)
Y,Ypred = evaluateModel(Models,Models0,Xtest,Ytest)

print((pd.DataFrame(Y)).shape)

