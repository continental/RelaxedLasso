"""
=====================================
  Cross-Validated RelaxedLassoLars
=====================================
An example  of : class relaxed_lasso.RelaxedLassoLarsCV
"""

from relaxed_lasso import RelaxedLassoLarsCV
from sklearn.datasets import make_regression

X, y, true_coefs = make_regression(n_samples=50,
                                   n_features=100,
                                   n_informative=5,
                                   noise=4.0,
                                   random_state=0,
                                   coef=True)

relassoCV = RelaxedLassoLarsCV(cv=3)  # 5 folds by default 
relassoCV.fit(X, y)

print("R-squared: ", relassoCV.score(X, y))

# Best parameters
print("Best Alpha: ", relassoCV.alpha_)
print("Best Theta: ", relassoCV.theta_)
