import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MAERegressor(BaseEstimator, RegressorMixin):
    """Lasso regression with MAE loss function."""
    def __init__(self, reg_lambda=1.0, solver='cplex'):
        self.lambda = lambda
        self.solver = solver

    def fit(self, X, y=None):
        # Check that X and y have the correct shape
        X, y = check_X_y(X, y, accept_sparse=True)

        if self.solver == 'cplex':
            
        
    
