import logging

from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

logger = logging.getLogger(__file__)


class MAERegressor(BaseEstimator, RegressorMixin):
    """Lasso regression with MAE loss function."""
    def __init__(self, reg_lambda=1.0, solver='cplex'):
        self.reg_lambda = reg_lambda
        self.solver = solver

    def fit(self, X, y=None):
        """Fit the model according to the given training data"""
        # Check that X and y have the correct shape
        X, y = check_X_y(X, y, accept_sparse=True)

        if self.solver == 'cplex':
            from .cplex_solver import cplex_solve
            intercept, coef = cplex_solve(X, y, self.reg_lambda)
        else:
            raise ValueError('MAERegressor only supports the cplex solver')
        self.intercept_, self.coef_ = intercept, coef

    def _decision_function(self, X):
        check_is_fitted(self, 'coef_')
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return (safe_sparse_dot(X, self.coef_.T, dense_output=True)
                + self.intercept_)

    def predict(self, X):
        return self._decision_function(X)
