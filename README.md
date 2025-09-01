# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries.

2.Load dataset (50_Startups.csv).

3.Separate features (X) and target (y).

4.Scale X and y using StandardScaler.

5.Print identification (Name & Reg. No.) and scaled values.

6.Build and train a LinearRegression model.

7.Scale new input, predict profit, and inverse transform the result.

8.Print final predicted profit.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: CHARUKESH S
RegisterNumber:  212224230044

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta
print("Name:CHARUKESH.S")
print("Register No:212224230044")
print(theta)
```
```
print("Name:CHARUKESH.S")
print("Register No:212224230044")
data=pd.read_csv("50_Startups.csv")
print(data.head())
```
```
X=(data.iloc[1:,:-2].values)
print(X)
```
```
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```
```
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print("Name:CHARUKESH.S")
print("Register No:212224230044")
print(X1_Scaled)
print(Y1_Scaled)
```
```
print("Name:CHARUKESH.S")
print("Register No:212224230044")
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
<img width="542" height="142" alt="image" src="https://github.com/user-attachments/assets/3f4a6f23-1f59-425b-828c-537403eeb30d" />
<img width="843" height="188" alt="image" src="https://github.com/user-attachments/assets/f7a04f87-6a64-4894-8894-ea2d7db12238" />
<img width="693" height="350" alt="image" src="https://github.com/user-attachments/assets/d289c9af-bca5-4041-a880-de0016fb069e" />
<img width="552" height="301" alt="image" src="https://github.com/user-attachments/assets/bac69681-485e-4e12-bc1b-3956a8003d91" />
<img width="667" height="447" alt="image" src="https://github.com/user-attachments/assets/52e3f1bd-699a-4b93-ae17-ee6df0f683e1" />
<img width="537" height="407" alt="image" src="https://github.com/user-attachments/assets/fb828362-be00-4f19-8cd7-1e498cbdd129" />
<img width="757" height="72" alt="image" src="https://github.com/user-attachments/assets/e0b8ab0a-e42c-4805-b15e-179fa3070892" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
