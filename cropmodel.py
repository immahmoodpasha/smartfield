import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb

# Define color array for plotting
colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
           '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
           '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']

# Load dataset
cropdf = pd.read_csv('Crop_recommendation.csv')

# Declare independent and target variables
X = cropdf.drop('label', axis=1)
y = cropdf['label']

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# Train LightGBM model
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))

# Training set accuracy
y_pred_train = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print('Training-set accuracy score: {0:0.4f}'.format(train_accuracy))

# Print scores on training and test set
print('Training set score: {:.4f}'.format(model.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(model.score(X_test, y_test)))

# View confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Accuracy: {:.4f}'.format(accuracy))
plt.show()

# View classification report
print(classification_report(y_test, y_pred))

# Example prediction
# new_data = [[98,35,18,23.79746068,74.82913698,6.252797547999999,91.76337172]]
# predicted_crop = model.predict(new_data)
# print('Predicted crop for new data:', predicted_crop)
import joblib

# Train your model as you did before
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'lgbm_model.pkl')
