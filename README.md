**Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored**

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict regression for marks by representing in a graph.
6.Compare graphs and hence linear regression is obtained for the given datas.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train, regressor.predict(x_train),color='blue') 
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train, regressor.predict(x_train),color='red')
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: shasmithaa sankar
RegisterNumber: 24900050

```

## Output:
![Screenshot 2024-11-14 082500](https://github.com/user-attachments/assets/d4d2c901-2c90-47e6-a29b-1d3884c4f933)
![Screenshot 2024-11-14 082512](https://github.com/user-attachments/assets/72b36a03-9e42-4ac1-9888-4d3b7e3bf267)
![Screenshot 2024-11-14 082534](https://github.com/user-attachments/assets/c8472d92-f3b9-4a08-874f-66a9e05963a1)
![Screenshot 2024-11-14 082601](https://github.com/user-attachments/assets/bf01314f-abd0-4d6b-99d7-e9bdefa2ddf4)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
