# Import required libraries

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import sklearn
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
# Importation des metriques d'évaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def lecture_csv(fichier):
    df = pd.read_csv(fichier, usecols =["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
    return df


df = lecture_csv("Pima_Indians_Diabetes.csv") 
#_______________DATA_SEPARATION_____________
print("\n> Séparation des données test et train ")
features = df.copy()
del features["Outcome"]
labels = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size = 0.20, shuffle=False)


# one hot encode outputs

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(500, activation='relu', input_dim=8))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(2, activation='softmax'))


# Compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=["accuracy"])


# build the model

history=model.fit(X_train, y_train, epochs=500,validation_data=(X_test, y_test))


# pred_train= model.predict(X_train)

scores = model.evaluate(X_train, y_train, verbose=0)

print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   

 

pred_test= model.predict(X_test)

scores2 = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))   

history_dict = history.history

print("KEYS : ", history_dict.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

