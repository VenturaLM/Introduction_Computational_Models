# Los nombre no están asignados

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

german = pd.read_csv('data/german.data-numeric', sep='\s+', header=None)

inputs_german = german.values[:, 0:-1]
outputs_german = german.values[:, -1]
train_inputs_german, test_inputs_german, train_outputs_german, test_outputs_german = train_test_split(
    inputs_german, outputs_german, test_size=0.4, random_state=42)

for nn in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=nn).fit(
        train_inputs_german, train_outputs_german)
    precisionTrain = knn.score(train_inputs_german, train_outputs_german)
    precisionTest = knn.score(test_inputs_german, test_outputs_german)
    print("%d vecinos: \tCCR train = %.2f%%, \tCCR test = %.2f%%" %
          (nn, precisionTrain*100, precisionTest*100))


reg = LogisticRegression(max_iter=1000).fit(
    train_inputs_german, train_outputs_german.ravel())
precisionTrain = reg.score(train_inputs_german, train_outputs_german)
precisionTest = reg.score(test_inputs_german, test_outputs_german)
print("Regresion logistica: \tCCR train = %.2f%%, \tCCR test = %.2f%%" %
      (precisionTrain*100, precisionTest*100))
