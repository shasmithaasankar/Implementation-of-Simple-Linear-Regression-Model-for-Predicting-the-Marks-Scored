# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students

2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored

3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis

4.Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b

5.for each data point calculate the difference between the actual and predicted marks

6.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error

7.Once the model parameters are optimized, use the final equation to predict marks for any new input data
 
 

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: shasmithaa sankar

RegisterNumber: 24900050
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
Head Values

![Screenshot 2024-08-16 154352-1](https://github.com/user-attachments/assets/f4ff0566-a2c7-4e52-9cd9-047aa451b049)

Tail Values

![Screenshot 2024-08-16 154419-1](https://github.com/user-attachments/assets/c6783304-cdba-4945-b9e3-0bf6f003a974)

X Values

![Screenshot 2024-08-16 152702](https://github.com/user-attachments/assets/d9236f1f-3fe3-49a4-9e03-38afa9500e2e)

y Values

![Screenshot 2024-08-16 153116-1](https://github.com/user-attachments/assets/3917d879-a092-4125-917f-ba9cdc2975af)

Predicted Values

![Screenshot 2024-08-16 161908](https://github.com/user-attachments/assets/c0410b60-3e91-49e5-9d8b-1eb6c0131986)

Actual Values

![Screenshot 2024-08-16 153301](https://github.com/user-attachments/assets/b6131f89-6ea3-43f0-9393-ce852d596ba6)

Testing Set

![download (8)](https://github.com/user-attachments/assets/9a1ab972-2e15-4795-838b-9d5c4c751f70)

Training Set

![download (7)-1](https://github.com/user-attachments/assets/5bc72231-c9f8-4bb8-9bb6-a072adbdf149)

MSE, MAE and RMSE

![Screenshot 2024-08-16 153958-1](https://github.com/user-attachments/assets/82d80c44-28c3-420a-af28-d5f8811e7d01)



## Result
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
