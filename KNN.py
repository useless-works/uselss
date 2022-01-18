import pandas as pd
from pandas import DataFrame
df_irisbd=pd.read_csv("iris.data")
print(df_irisbd)
x=df_irisbd.iloc[:, :-1].values
y=df_irisbd.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=100)
print(y_test)
print("****************")
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
predicted=model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predicted))
print("*****************")
print(classification_report(y_test,predicted))
