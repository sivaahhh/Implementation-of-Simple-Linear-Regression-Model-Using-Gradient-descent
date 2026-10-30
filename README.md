# Implementation-of-Simple-Linear-Regression-Model-Using-Gradient-descent

# AIM:

To write a program to predict the profit of a city using the linear regression model with gradient descent.

# Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm:

1.Import the required library and read the dataframe. 2.Write a function compute Cost to generate the cost function. 3.Perform iterations of gradient steps with learning rate. 4.Plot the Cost function using Gradient Descent and generate the required graph.

# Program:

/*
Program to implement the linear regression using gradient descent.

Developed by: SIVA SAKTHI A

RegisterNumber: 212224223005

/*

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
```
data=pd.read_csv("/content/exp_3_50_Startups.csv")
x=data["R&D Spend"].values
y=data["Profit"].values
```
```
x_mean=np.mean(x)
x_std=np.std(x)
x=(x-x_mean)/x_std
```
```
w=0.0
b=0.0
alpha=0.01
epochs=100
n=len(x)
```
```
losses=[]
```
```
for i in range(epochs):
    y_hat=w*x+b
    loss=np.mean((y_hat-y)**2)
    losses.append(loss)
    
    dw=(2/n)*np.sum((y_hat-y)*x)
    db=(2/n)*np.sum(y_hat-y)
    
    w-=alpha*dw
    b-=alpha*db
```
```
plt.figure(figsize=(12,5))
```
```
plt.subplot(1,2,1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss(MSE)")
plt.title("Loss vs Iterations")
```
```

plt.subplot(1,2,2)
plt.scatter(x,y)
x_sorted=np.argsort(x)
plt.plot(x[x_sorted],(w*x+b)[x_sorted],color="red")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")
```
```

plt.tight_layout()
plt.show()
```
```

print(f"Final weight (w): {w}")
print(f"Final bias (b): {b}")
```
# OUTPUT:

<img width="545" height="71" alt="image" src="https://github.com/user-attachments/assets/d2538c90-14ef-4db6-b130-6fa4a3e22326" />
<img width="585" height="626" alt="image" src="https://github.com/user-attachments/assets/ad8588b1-a8ef-4e6c-83f6-3f583654dd05" />
<img width="652" height="617" alt="image" src="https://github.com/user-attachments/assets/bab29aa3-281c-4f7e-afca-100fccda17a6" />
<img width="518" height="45" alt="image" src="https://github.com/user-attachments/assets/d0d225ea-22be-4864-ab06-2fd646d189a2" />
<img width="632" height="73" alt="image" src="https://github.com/user-attachments/assets/714fde4f-0075-475f-af17-1abb4e1dcfac" />

# Result:

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
