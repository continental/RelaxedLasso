"""
==========================
  RelaxedLasso Regressor
==========================
An example  of : class relaxed_lasso.RelaxedLasso
"""

from relaxed_lasso import RelaxedLasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=20,
                       n_features=5,
                       n_informative=3,
                       noise=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

relasso=RelaxedLasso()
relasso.fit(X_train, y_train)
    
print("Coefs:",relasso.coef_[relasso.coef_!=0])
print("Pred:", relasso.predict(X_test))
print("Actual obs:", y_test)
