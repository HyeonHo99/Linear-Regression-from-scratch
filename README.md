# Linear-Regression-from-scratch
Numpy implementation of Linear Regression (without using explicit ML libraries)


## Types of Regression Models
<div>
  <div>
    <img src="imgs/types.jpg" width="600" height="480"/>
  </div>
  <div>
    ✔️ Here we build a <b>Multiple Linear Regression Model</b> which assumes input features are multiple (input data with single feature is also applicable) <br>
        <img src="imgs/multiple_linear_regression.PNG" width="650" height="80"/>
  </div>
</div>

## Error function (Loss function) : Least Squares Error
<div>
Least Squares Error aims to minimize the sum of squared residuals between actual values and predicted values <br>
Formulation is defined as below <br>
<img src="imgs/mse.PNG" width="300" height="60"/><br>
In the case 'Simple Linear Regression' (one feature for one data), it can be indicated as <br>
<img src="imgs/mse_simple.PNG" width="300" height="100"/>
</div>

## Solving Optimization
### 1. Analytical Solution (using Normal Equation)
<div>
  Let's define <br>
  <b>X</b>:&nbsp;&nbsp; Input data &nbsp;&nbsp;Shape: n x (d+1) <br>
  <b>w</b>:&nbsp;&nbsp; weights &nbsp;&nbsp;Shape: (d+1) x 1 <br>
  <b>y&#770;</b>:&nbsp;&nbsp; predicted value &nbsp;&nbsp;Shape: nx1 <br>
  <b>y</b>:&nbsp;&nbsp; ground truth &nbsp;&nbsp;Shape: nx1 <br>
  <img src="imgs/vectors_def.PNG" width="500" height="250"/>
</div>
<div>
  In the case of <b>Least Squares Error</b>, analytical solution is derived as below. <br>
  <img src="imgs/normal_equation.PNG" width="400" height="300"/>
</div>

```python
def analytic_solution(self, x, y):
    x_T = np.transpose(x)
    inverse = np.linalg.inv(np.dot(x_T,x))
    self.W = np.dot(inverse,np.dot(x_T,y))
```
#### General Derivation
  <img src="imgs/general.PNG" width="450" height="250"/><br>
#### Detailed Derivation
  <img src="imgs/detail.PNG" width="500" height="250"/><br>

### 2. Numerical Solution (Batch Gradient Descent)
<div>
  Compute gradient using <b>full training samples</b><br>
  
  And Update <b>W</b>(weights)<br>
</div>
