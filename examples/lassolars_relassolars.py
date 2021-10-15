"""
=============================================
  RelaxedLassoLarsCV vs LassoLars RegressorCV
=============================================
An example  of : class relaxed_lasso.RelaxedLassoLarsCV
"""

import numpy as np
from relaxed_lasso import RelaxedLassoLarsCV
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split

# Create highly colinear dataset for regression.
# The data set has 50 samples and 100 features
#  out of which 6 are used to compute the target
#  but only 3 have coefficients greater than the noise

mu = np.repeat(0, 100)
dists = np.arange(100)
powers = [[np.abs(i-j) for j in dists] for i in dists]
r = np.power(.5, powers)
X = np.random.multivariate_normal(mu, r, size=50)
y = 7*X[:, 0] + \
    5*X[:, 10] + \
    3*X[:, 20] + \
    1*X[:, 30] + \
    .5*X[:, 40] + \
    .2*X[:, 50] + \
    np.random.normal(0, 2, 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                    random_state=0)

relasso = RelaxedLassoLarsCV().fit(X_train, y_train)
print(f"Relaxed lasso score on test set: {relasso.score(X_test, y_test)}")

lasso = LassoLarsCV().fit(X_train, y_train)
print(f"Lasso score on test set: {lasso.score(X_test, y_test)}")

print(f"""True number of predictors: 6
    Predictors retained by relaxed lasso: {np.count_nonzero(relasso.coef_)}
    Predictors retained by  lasso: {np.count_nonzero(lasso.coef_)}""")
