#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load the csv file
df = pd.read_csv("iris.csv")

# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Convert labels to numeric categories
y = pd.factorize(y)[0]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[2]:


# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=50)
rf_classifier.fit(X_train_scaled, y_train)
rf_predictions = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)


# In[3]:


# Logistic Regression
lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_scaled, y_train)
lr_predictions = lr_classifier.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Logistic Regression Accuracy:", lr_accuracy)


# In[4]:


# Support Vector Machine
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train_scaled, y_train)
svm_predictions = svm_classifier.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("Support Vector Machine Accuracy:", svm_accuracy)


# In[5]:


# K-Nearest Neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)
knn_predictions = knn_classifier.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)


# In[6]:


# Deep Learning
model_dl = Sequential([
    Dense(64, activation='relu', input_shape=(4,)),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])
model_dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_dl.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
dl_loss, dl_accuracy = model_dl.evaluate(X_test_scaled, y_test)
print("Deep Learning Accuracy:", dl_accuracy)


# In[7]:


# Save the models


with open("logistic_regression_model.pkl", "wb") as file:
    pickle.dump(lr_classifier, file)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

