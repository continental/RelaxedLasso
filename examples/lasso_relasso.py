"""
=============================================
  RelaxedLassoLarsCV vs LassoLars RegressorCV
=============================================
An example  of : class relaxed_lasso.RelaxedLassoLarsCV
"""

import numpy as np
from relaxed_lasso import RelaxedLassoLarsCV
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split

X, y, true_coefs = make_regression(n_samples=25,
                                   n_features=100,
                                   n_informative=5,
                                   noise=4.0,
                                   random_state=0,
                                   coef=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=0)

relasso = RelaxedLassoLarsCV().fit(X_train, y_train)
print(f"Relaxed lasso score on test set: {relasso.score(X_test, y_test)}")

lasso = LassoLarsCV().fit(X_train, y_train)
print(f"Lasso score on test set: {lasso.score(X_test, y_test)}")

print(f"""True number of predictors: {np.count_nonzero(true_coefs)}
    Predictors retained by relaxed lasso: {np.count_nonzero(relasso.coef_)}
    Predictors retained by  lasso: {np.count_nonzero(lasso.coef_)}""")
