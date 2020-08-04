"""
===============================
  RelaxedLassoLars Regressor
===============================
An example  of : class relaxed_lasso.RelaxedLassoLars
"""

from relaxed_lasso import RelaxedLassoLars
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=50,
                       n_features=100,
                       n_informative=5,
                       noise=4.0,
                       random_state=0)

relasso = RelaxedLassoLars()
relasso.fit(X, y)
relasso.predict(X)
