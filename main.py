from matplotlib import pyplot
import pandas as pd
import numpy as np

fileXtrain = 'X_train.txt'
fileYtrain = 'y_train.txt'
fileXtest = 'X_test.txt'
fileYtest = 'y_test.txt'

def binarizeData(Y,ref):
  for i in range(len(Y)):
    if Y[i] == ref:
      Y[i] = 1
    else:
      Y[i] = 0
  return Y

def correlationMatrix(X):
  correlations = X.corr()
  fig = pyplot.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(correlations, vmin=-1, vmax=1)
  fig.colorbar(cax)
  pyplot.show()

def setNewData(X,Y,ref):
  Xdata = pd.concat([X,pd.DataFrame(Y)],axis=1)
  alpha=0.3
  f1 = 89
  f2 = 95
  X = X.values.tolist()
  for i in range(len(Y)):
    if Y[i] == 1:
      pyplot.scatter(X[i][f1],X[i][f2],color='blue',alpha=alpha)
    if Y[i] == 2:
      pyplot.scatter(X[i][f1],X[i][f2],color='red',alpha=alpha)
    if Y[i] == 3:
      pyplot.scatter(X[i][f1],X[i][f2],color='green',alpha=alpha)
    if Y[i] == 4:
      pyplot.scatter(X[i][f1],X[i][f2],color='yellow',alpha=alpha)
    if Y[i] == 5:
      pyplot.scatter(X[i][f1],X[i][f2],color='purple',alpha=alpha)
    if Y[i] == 6:
      pyplot.scatter(X[i][f1],X[i][f2],color='black',alpha=alpha)
  pyplot.show()
  plotTwoFeatures(X,Y,f1,f2)
  return X,Y

def plotTwoFeatures(X,Y,f1,f2):
  alpha=0.3
  for i in range(len(Y)):
    if Y[i] == 1:
      pyplot.scatter(X[i][f1],X[i][f2],color='blue',alpha=alpha)
    if Y[i] == 2:
      pyplot.scatter(X[i][f1],X[i][f2],color='red',alpha=alpha)
    if Y[i] == 3:
      pyplot.scatter(X[i][f1],X[i][f2],color='green',alpha=alpha)
    if Y[i] == 4:
      pyplot.scatter(X[i][f1],X[i][f2],color='yellow',alpha=alpha)
    if Y[i] == 5:
      pyplot.scatter(X[i][f1],X[i][f2],color='purple',alpha=alpha)
    if Y[i] == 6:
      pyplot.scatter(X[i][f1],X[i][f2],color='black',alpha=alpha)
  pyplot.show()


X = pd.read_csv('X_train.csv',sep=' ')
Y = pd.read_csv('y_train.csv',sep=' ').values.tolist()
setNewData(X,Y,1)

