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

## Error function (Loss function) : Mean Squared Error
<div>
MSE aims to minimize the sum of squared residuals between actual values and predicted values <br>
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
</div>
