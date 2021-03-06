import logging
import numpy as np

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

        # Get feature importances: coeffients scaled by median absolute
        # deviations of corresponding variables
        medians = np.median(X, axis=0)
        medians.shape = (1, len(medians))
        AD = abs(X - medians)
        MAD = np.median(AD, axis=0)
        self.feature_importance_ = MAD*coef

        median_response = np.median(y)
        self.median_response_ = median_response

    def _decision_function(self, X):
        check_is_fitted(self, 'coef_')
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return (safe_sparse_dot(X, self.coef_.T, dense_output=True)
                + self.intercept_)

    def predict(self, X):
        return self._decision_function(X)

    def score(self, X, y):
        AD = mean_absolute_error(self.predict(X), y)
        AD0 = mean_absolute_error(np.full(len(y), self.median_response_), y)
        if AD0 == 0:
            R1 = 0.
        else:
            R1 = (AD0 - AD)/AD0
        return R1
