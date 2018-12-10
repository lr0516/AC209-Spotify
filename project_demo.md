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
from Bambanta import AutoDiff
from time import clock
```




```python
def print_jac(val, jac, t):
    print('Function Value: ', val)
    print('Function Jacobian: ')
    print(jac)
    print('---------------------------------')
    print('Time Used: ', t)

def print_grad(val,g1,g2,g3,g4,t):
    print('Function Value: ', val)
    print('df/dx1: ', g1)
    print('df/dx2: ', g2)
    print('df/dx3: ', g3)
    print('df/dx4: ', g4)
    print('---------------------------------')
    print('Time Used: ', t)

def print_opti(sol,n,t):
    print('Optimized solution: ',sol)
    print('Number of Steps Used: ',n)
    print('---------------------------------')
    print('Time Used: ', t)
```




```python
## Case 1.1 Multivariate with fAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_f(
    [np.pi/2,1,3,1])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
t = clock() - t0

print_jac(f.get_val(),f.get_jac(),t)
```


    Function Value:  5.0
    Function Jacobian: 
    [6.123234e-17 3.000000e+00 1.000000e+00 4.000000e+00]
    ---------------------------------
    Time Used:  0.0006650000000000267




```python
## Case 1.2 Multivariate with rAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_r(
    [np.pi/2,1,3,1])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
f.outer()
t = clock() - t0

print_grad(f.get_val(),x1.get_grad(),x2.get_grad(),
          x3.get_grad(),x4.get_grad(),t)
```


    Function Value:  5.0
    df/dx1:  6.123233995736766e-17
    df/dx2:  3.0
    df/dx3:  1.0
    df/dx4:  4.0
    ---------------------------------
    Time Used:  0.0007670000000000732




```python
## Case 2.1 2-D Variables with fAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_f(
    [[np.pi/2,0],[1,2],[3,4],[1,2]])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
t = clock() - t0
print_jac(f.get_val(),f.get_jac(),t)
```


    Function Value:  [ 5. 24.]
    Function Jacobian: 
    [[6.123234e-17 3.000000e+00 1.000000e+00 4.000000e+00]
     [1.000000e+00 4.000000e+00 2.000000e+00 3.200000e+01]]
    ---------------------------------
    Time Used:  0.001786999999999983




```python
## Case 2.2 2-D Variables with rAD
t0 = clock()
x1,x2,x3,x4 = AutoDiff.create_r(
    [[np.pi/2,0],[1,2],[3,4],[1,2]])
f = AutoDiff.sin(x1) + x2*x3 + x4**4
f.outer()
t = clock() - t0

print_grad(f.get_val(),x1.get_grad(),x2.get_grad(),
          x3.get_grad(),x4.get_grad(),t)
```


    Function Value:  [ 5. 24.]
    df/dx1:  [6.123234e-17 1.000000e+00]
    df/dx2:  [3. 4.]
    df/dx3:  [1. 2.]
    df/dx4:  [ 4. 32.]
    ---------------------------------
    Time Used:  0.0006180000000000074




```python
## Case 3.1 2-D Functions with fAD
t0 = clock()
x1,x2,x3 = AutoDiff.create_f([1,np.pi/4,2])
f1 = x1*x2 + x3**4
f2 = -AutoDiff.log(x1) + AutoDiff.cos(x2) * x3
f = AutoDiff.stack_f([f1,f2])
t = clock() - t0
print_jac(f.get_val(),f.get_jac(),t)
```


    Function Value:  [16.78539816  1.41421356]
    Function Jacobian: 
    [[ 0.78539816  1.         32.        ]
     [-1.         -1.41421356  0.70710678]]
    ---------------------------------
    Time Used:  0.002595000000000014




```python
## Case 3.2 2-D Functions with rAD
t0 = clock()
def f1(x1,x2,x3): 
    return x1*x2 + x3**4
def f2(x1,x2,x3): 
    return -AutoDiff.log(x1) + AutoDiff.cos(x2) * x3
f_val, f_jac = AutoDiff.stack_r([1,np.pi/4,2],[f1,f2])
t = clock() - t0

print_jac(f_val,f_jac,t)
```


    Function Value:  [16.78539816  1.41421356]
    Function Jacobian: 
    [[ 0.78539816  1.         32.        ]
     [-1.         -1.41421356  0.70710678]]
    ---------------------------------
    Time Used:  0.002151000000000014


#### Rosenbrock: $$f(x,y) = 100(y-x^2)^2+(1-x)^2 $$



```python
from scipy.optimize import line_search
from numpy.linalg import norm
```




```python
## Case 4.1 Steepest Descent with fAD
t0 = clock()

def R(X): 
    return 100*(X[1]-X[0]**2)**2 + (1-X[0])**2

def G(X):     
    x,y = AutoDiff.create_f([X[0],X[1]])
    f = R([x,y])
    return f.get_jac()

x,y,dX,step_count = 0,1,[1,1],0
while norm(dX) >= 1e-8 and step_count < 2000:
    step_count += 1
    s = -G([x,y])
    eta = line_search(R, G, [x,y], s)[0]
    dX = eta*s
    x,y = [x,y]+dX
t = clock() - t0

print_opti([x,y],step_count,t)
```


    Optimized solution:  [0.9999995671204049, 0.9999991324744798]
    Number of Steps Used:  1571
    ---------------------------------
    Time Used:  1.3865530000000001




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

x,y,dX,step_count = 0,1,[1,1],0
while norm(dX) >= 1e-8 and step_count < 2000:
    step_count += 1
    s = -G([x,y])
    eta = line_search(R, G, [x,y], s)[0]
    dX = eta*s
    x,y = [x,y]+dX
t = clock() - t0

print_opti([x,y],step_count,t)
```


    Optimized solution:  [0.9999995671190041, 0.9999991324716722]
    Number of Steps Used:  1571
    ---------------------------------
    Time Used:  1.0821429999999999




```python

```

