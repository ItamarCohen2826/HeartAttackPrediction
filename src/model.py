from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./data/data.csv')
x = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
model = LogisticRegression()
history = model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_testArray = df.to_numpy(y_test)
print('Prediction: ')
print(y_pred)
print(' - - - - - - - - - - - - - - - - - - - - -')
print('Testing:')
print(y_test)
print(' - - - - - - - - - - - - - - - - - - - - - - -')
print('f1 Score:')
f_score = f1_score(y_test, y_pred)
print(f_score)
print(' - - - - - - - - - - - - - - - - - - - - -')
print('Class Score:')

class_score = classification_report(y_test, y_pred)
print(class_score)
print(' - - - - - - - - - - - - - - - - - - - - -')
# print(y_testArray)
val_array = np.reshape([45,0,615,1,55,0,222000,0.8,141,0,0,257], (1, -1))
prediction = model.predict(val_array)
print(prediction)
""" for i in range(len(y_pred)):
        for j in range(len(y_testArray)):
            evaluationArray = y_pred[i] - y_testArray[j]
print(evaluationArray)  """