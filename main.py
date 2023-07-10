from matplotlib import pyplot as plt
from pandas import read_csv as r
import pandas as pd
fileXtrain = 'X_train.txt'
fileYtrain = 'y_train.txt'
fileXtest = 'X_test.txt'
fileYtest = 'y_test.txt'
xTrain = pd.read_csv('X_train.txt',sep=' ')
xTrain.to_csv('X_train.csv', index=None)
xTrain.hist()
plt.show()