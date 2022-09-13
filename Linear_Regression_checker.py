import numpy as np
from models.LinearRegression import LinearRegression

from utils import optimizer, RMSE


np.random.seed(10)

Dataset = np.loadtxt('data/check_data.txt')
x_data, y_data = Dataset[:, :-1], Dataset[:, -1]

_epoch = 100
_batch_size = 5
_lr = 0.01
_optim = 'SGD'

#======================================================================================================

print('='*20, 'Batch Gradient Numerical Solution Test', '='*20)

Numeric_model = LinearRegression(num_features=x_data.shape[1])
optimizer = optimizer(_optim)
print('Initial weight: \n', Numeric_model.W.reshape(-1))
print()

Numeric_model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer, batch_gradient=True)
print('Trained weight: \n', Numeric_model.W)
print()

# model evaluation
inference = Numeric_model.eval(x_data)

# Error calculation
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f \n' % error)

"""
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight:
 [ 18.9987147  -11.9176875   19.99074236  25.70189935]

RMSE on Train Data : 30.4890 
"""

#======================================================================================================

#======================================================================================================

print('='*20, 'Mini-Batch Stochastic Gradient Numerical Solution Test', '='*20)

Numeric_model = LinearRegression(num_features=x_data.shape[1])
print('Initial weight: \n', Numeric_model.W.reshape(-1))
print()

Numeric_model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optimizer, batch_gradient=False)
print('Trained weight: \n', Numeric_model.W)
print()

# model evaluation
inference = Numeric_model.eval(x_data)

# Error calculation
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f \n' % error)

"""
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight:
 [ 22.15868831 -11.56430191  31.24868975  34.05022129]

RMSE on Train Data : 22.1719
"""

#======================================================================================================

print('='*20, 'Analytic Solution Test', '='*20)

Analytic_model = LinearRegression(num_features=x_data.shape[1])
print('Initial weight: \n', Analytic_model.W.reshape(-1))
print()

Analytic_model.analytic_solution(x=x_data, y=y_data)
print('Trained weight: \n', Analytic_model.W.reshape(-1))
print()

# model evaluation
inference = Analytic_model.eval(x_data)

# Error calculation
error = RMSE(inference, y_data)
print('RMSE on Test Data : %.4f \n' % error)

"""
You should get results as:

Initial weight:
 [0. 0. 0. 0.]

Trained weight:
 [ 0.76694866 20.82887592 54.13176192 76.77324443]

RMSE on Test Data : 0.3732
"""
#======================================================================================================