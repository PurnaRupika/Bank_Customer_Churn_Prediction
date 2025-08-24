import tensorflow as tf


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
%matplotlib inline

df=pd.read_csv("/content/Churn_Modelling.csv")
df.sample(5)

df.drop('RowNumber',axis='columns',inplace=True)
df.drop('CustomerId',axis='columns',inplace=True)
df.drop('Surname',axis='columns',inplace=True)

df.dtypes

tenure_Exited_No=df[df.Exited==0].Tenure
tenure_Exited_Yes=df[df.Exited==1].Tenure

plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Bank Customer Churn Prediction Visualization")

plt.hist([tenure_Exited_Yes,tenure_Exited_No],color=['green','red'],label=['Tenure=1','Tenure=0'])
plt.legend()

for column in df:
    print(f'{column}: {df[column].unique()}')

def print_unique_col_values(df):
    for column in df:
         if df[column].dtypes=='object':
             print(f'{column}: {df[column].unique()}')

print_unique_col_values(df)

df['Gender'].replace({'Female':1,'Male':0},inplace=True)
df['Gender'].unique()

#one hot encoder
df1 = pd.get_dummies(data=df,columns=['Geography'],dtype=int)

df1.columns

df1.sample(5)

df1.dtypes

#scaling the columns between 0 and 1
cols_to_scale=['Tenure', 'Balance', 'EstimatedSalary','CreditScore']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df1[cols_to_scale] = scaler.fit_transform(df1[cols_to_scale])

df1.sample(5)

X = df1.drop('Exited',axis='columns')
y = df1['Exited']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

X_train.shape

X_test.shape

X_train[:10]

len(X_train.columns)

from tensorflow import keras

model = keras.Sequential([
 keras.layers.Dense(12,input_shape=(12,), activation='relu') ,
  keras.layers.Dense(1,activation='sigmoid') ,
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train,y_train,epochs=100)

model.evaluate(X_test,y_test)

y_predict = model.predict(X_test)
y_predict[:5]

y_new_predict = []
for element in y_predict:
  if element > 0.5:
    y_new_predict.append(1)
  else:
      y_new_predict.append(0)

y_new_predict[:15]

from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_new_predict))

import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_new_predict)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')



