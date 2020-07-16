"""
===============================
  RelaxedLassoLars Regressor
===============================
An example  of : class relaxed_lasso.RelaxedLassoLars
"""

from relaxed_lasso import RelaxedLassoLars
from sklearn.datasets import make_regression

X, y, true_coefs = make_regression(n_samples=50,
                                   n_features=1000,
                                   n_informative=5,
                                   noise=4.0,
                                   random_state=0,
                                   coef=True)

relasso = RelaxedLassoLars()
relasso.fit(X, y)
relasso.predict(X)