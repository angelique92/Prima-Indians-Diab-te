#_______________IMPORTS_____________
from keras import layers, Model, optimizers, regularizers
# Importation pour manipuler les fichier csv
import pandas as pd
# Importation des metriques d'évaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Importation permettant de séparé notre jeu de données
from sklearn.model_selection import train_test_split

# importation lié aux divers calcul matricielle
import numpy as np
import pandas as pd
import csv
# Scikit-learn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt 
def lecture_csv(fichier):
    df = pd.read_csv(fichier, usecols =["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"])
    return df

liste=[]
def RDN(X_train,X_test,y_train,y_test, n1,n2):
    #Preparation du Réseau de neurones avec Keras
    model = Sequential()
    model.add(Dense(n1, input_dim=8, activation="relu"))
    model.add(Dense(n2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    #Compile le model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #Entraine le model
    history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

    model.save("t2_tp1.h5")

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Predict all test data
    prediction = model.predict(X_test)

    # Take the best classe (hight score) for all prediction
    bestPred = [1 if pred >= 0.5 else 0 for pred in prediction]

    # Calculate accuracy : How many prediction are good ?
    acc = accuracy_score(bestPred, y_test)

    # Calculate precision : Number of correct prediction for this class / total of predictions for this class
    precision = precision_score(bestPred, y_test)

    # Calculate recall : Number of correct prediction  / total element of this class
    recall = recall_score(bestPred, y_test)

    # Relation beetwen precision and recall
    f1Score = f1_score(bestPred, y_test)

    print("\nAccuracy:", acc*100, "\nPrecision :", precision*100, "\nRecall", recall*100, "\nF1 score", f1Score*100)

    liste.append([acc*100,precision*100,recall*100,f1Score*100])

if __name__ == '__main__':

    #_______________LOAD_DATA_____________
    print("Lecture du fichier")
    df=lecture_csv("Pima_Indians_Diabetes.csv")

    
    #_______________IMPROVE_DATA_____________
    # print("SHAPE AVANT :", df)
    columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in columns:
        df[col].replace(0, np.NaN, inplace=True)
    print("SHAPE AVANT :", df)
    # Drop rows with missing values
    df.dropna(inplace=True)
    print("SHAPE APRES :", df)
    # print(df["Outcome"])


    #_______________DATA_SEPARATION_____________
    print("\n> Séparation des données test et train ")
    features = df.copy()
    del features["Outcome"]
    labels = df["Outcome"]

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    StandardScaler(copy=True, with_mean=True, with_std=True)

    # Transform and display the training data
    X_standardized = scaler.transform(features)

    data = pd.DataFrame(X_standardized)

    X_train, X_test, y_train, y_test = train_test_split(data, labels,test_size = 0.20, shuffle=False)

    # ____________________MODELES_CREATION____________
    # RDN(X_train, X_test, y_train, y_test,2,6)
    # RDN(X_train, X_test, y_train, y_test,12,8)
    
    # RDN(X_train, X_test, y_train, y_test,200,40)
    RDN(X_train, X_test, y_train, y_test,1000,400)



    print(liste)
