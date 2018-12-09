---
title: Project Demo
notebook: project_demo.ipynb
---

## Contents
{:.no_toc}
*  
{: toc}




```python
import numpy as np
from project_repo.AutoDiff import AutoDiff
from time import clock
```




```python
## Case 1.1 Multivariate with fAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_f([np.pi/2,1,3,1])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
t = clock() - t0
print('Function Value: ', f.get_val())
print('Function Jacobian: ', f.get_jac())
print('---------------------------------')
print('Time Used: ', t)
```


    Function Value:  5.0
    Function Jacobian:  [6.123234e-17 3.000000e+00 1.000000e+00 4.000000e+00]
    ---------------------------------
    Time used:  0.0016649999999999165




```python
## Case 1.2 Multivariate with rAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_r([np.pi/2,1,3,1])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
f.outer()
t = clock() - t0
print('Function Value: ', f.get_val())
print('df/dx1: ', x1.get_grad())
print('df/dx2: ', x2.get_grad())
print('df/dx3: ', x3.get_grad())
print('df/dx4: ', x4.get_grad())
print('---------------------------------')
print('Time Used: ', t)
```


    Function Value:  5.0
    df/dx1:  6.123233995736766e-17
    df/dx2:  3.0
    df/dx3:  1.0
    df/dx4:  4.0
    ---------------------------------
    Time used:  0.0018039999999999168




```python
## Case 2.1 2-D Variables with fAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_f([[np.pi/2,0],[1,2],[3,4],[1,2]])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
t = clock() - t0
print('Function Value: ', f.get_val())
print('Function Jacobian: ', f.get_jac())
print('---------------------------------')
print('Time Used: ', t)
```


    Function Value:  [ 5. 24.]
    Function Jacobian:  [[6.123234e-17 3.000000e+00 1.000000e+00 4.000000e+00]
     [1.000000e+00 4.000000e+00 2.000000e+00 3.200000e+01]]
    ---------------------------------
    Time used:  0.0029799999999999827




```python
## Case 2.2 2-D Variables with rAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_r([[np.pi/2,0],[1,2],[3,4],[1,2]])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
f.outer()
t = clock() - t0
print('Function Value: ', f.get_val())
print('df/dx1: ', x1.get_grad())
print('df/dx2: ', x2.get_grad())
print('df/dx3: ', x3.get_grad())
print('df/dx4: ', x4.get_grad())
print('---------------------------------')
print('Time Used: ', t)
```


    Function Value:  [ 5. 24.]
    df/dx1:  [6.123234e-17 1.000000e+00]
    df/dx2:  [3. 4.]
    df/dx3:  [1. 2.]
    df/dx4:  [ 4. 32.]
    ---------------------------------
    Time used:  0.0022719999999999407




```python
## Case 3.1 2-D Functions with fAD
t0 = clock()
x1,x2,x3 = AutoDiff.create_f([1,np.pi/4,2])
f1 = x1*x2 + x3**4
f2 = -AutoDiff.log(x1) + AutoDiff.cos(x2) * x3
f = AutoDiff.stack_f([f1,f2])
t = clock() - t0
print('Function Value: ', f.get_val())
print('Function Jacobian: ', f.get_jac())
print('---------------------------------')
print('Time Used: ', t)
```


    Function Value:  [16.78539816  1.41421356]
    Function Jacobian:  [[ 0.78539816  1.         32.        ]
     [-1.         -1.41421356  0.70710678]]
    ---------------------------------
    Time used:  0.002715000000000023




```python
## Case 3.2 2-D Functions with rAD
t0 = clock()
def f1(x1,x2,x3): return x1*x2 + x3**4
def f2(x1,x2,x3): return -AutoDiff.log(x1) + AutoDiff.cos(x2) * x3
f_val, f_jac = AutoDiff.stack_r([1,np.pi/4,2],[f1,f2])
t = clock() - t0
print('Function Value: ', f_val)
print('Function Jacobian: ', f_jac)
print('---------------------------------')
print('Time Used: ', t)
```


    Function Value:  [16.78539816  1.41421356]
    Function Jacobian:  [[ 0.78539816  1.         32.        ]
     [-1.         -1.41421356  0.70710678]]
    ---------------------------------
    Time used:  0.001711000000000018


#### Rosenbrock: $$f(x,y) = 100(y-x^2)^2+(1-x)^2 $$



```python
## Case 4.1 Steepest Descent with fAD
from scipy.optimize import line_search
t0 = clock()

def R(X): 
    return 100*(X[1]-X[0]**2)**2 + (1-X[0])**2

def G(X):     
    x,y = AutoDiff.create_f([X[0],X[1]])
    f = R([x,y])
    return f.get_jac()

x,y = 0,1
dX = [1,1] # to ensure that norm(dX) >= 1e-8
step_count = 0
while np.linalg.norm(dX) >= 1e-8 and step_count < 2000:
    step_count += 1
    s = -G([x,y])
    eta = line_search(R, G, [x,y], s)[0]
    x,y = [x,y]+eta*s
    
t = clock() - t0
print('Optimized solution: ',[x,y])
print('Number of Steps Used: ',step_count)
print('---------------------------------')
print('Time Used: ', t)
```


    Optimized solution:  [0.9999999999999148, 0.9999999999998291]
    Number of Steps Used:  2000
    ---------------------------------
    Time Used:  1.4958799999999997




```python
## Case 4.2 Steepest Descent with rAD
t0 = clock()

def R(X): 
    return 100*(X[1]-X[0]**2)**2 + (1-X[0])**2

def G(X):
    x,y = AutoDiff.create_r(X)
    f = R([x,y])
    f.outer()
    return np.array([x.get_grad(), y.get_grad()])

x,y = 0,1
dX = [1,1] # to ensure that norm(dX) >= 1e-8
step_count = 0
while np.linalg.norm(dX) >= 1e-8 and step_count < 2000:
    step_count += 1
    s = -G([x,y])
    eta = line_search(R, G, [x,y], s)[0]
    dX = eta*s
    x,y = [x,y]+dX

t = clock() - t0
print('Optimized solution: ',[x,y])
print('Number of Steps Used: ',step_count)
print('---------------------------------')
print('Time Used: ', t)
```


    Optimized solution:  [0.9999995671190041, 0.9999991324716722]
    Number of Steps Used:  1571
    ---------------------------------
    Time Used:  1.159984999999999

