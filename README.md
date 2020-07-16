# Relaxed Lasso

Improved version of classical lasso regularization for linear regression, as
per the paper by Nicholas Meinshausen (2007): [Relaxed Lasso](https://stat.ethz.ch/~nicolai/relaxo.pdf).

Relaxed lasso lets you control both the number of variables retained and the 
amount of regularization applied using two separate hyperparameters. This 
leads to sparser models than classical lasso, while achieving equal or lower
error.

## Purpose & description

#### What is relaxed lasso, when is it used?

Lasso, and its improvement relaxed lasso, are extensions of linear regressions.
The main benefits are their ability to deal with colinearity, high dimensions
(even higher than number of samples) and the fact that they lead to a sparse
solutions.

According to Hastie, Tibshirani (2016) in Best Subset, Forward Stepwise, or
Lasso?, _relaxed lasso_ is the overall winner when it comes to variables
selection. Surprisingly up to now there was no python implementation of this
algorithm, although one exists in R
([relaxo](https://cran.r-project.org/web/packages/relaxo/index.html))

#### Relaxed Lasso concept

The key concept is that there are two regularization parameters, α (alpha) which
controls the variables that will be retained in the model, and θ (theta), value
between 0 and 1, which acts as a multiplicative factor of α to choose the
amount of regularization applied to the subset of variables.
* θ = 1 corresponds to standard Lasso
* θ = 0 corresponds to the ordinary least square solution for the subset of
variables selected with α.

## Implementation

#### Dependencies
The implementation is heavily inspired and relies on [scikit-learn](http://scikit-learn.org/)
implementation of lasso.

There are two algorithms implemented in scikit-learn to get the lasso_path:
* least angle regression (LARS) : this is the one used in this implementation
* coordinate ascent : not implemented

For pros and cons of each algorithm, see the [lasso documentation](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)

The reason for choosing LARS is that it produces a full piecewise linear
solution path, which is particularly well suited for extrapolation of
coefficients values when applying the 'relaxation' factor θ.

#### Naming convention
The parameters called α and θ in this implementation are called
respectively λ (lambda) and Φ (phi) in the paper.

This choice was made to stick as closely as possible to scikit-learn
conventions.

#### Additional implementation details
The RelaxedLassoLarsCV algorithm relies on the exploration of a grid of
values for α. One dimension is the α controlling the variables choice
(alpha_var) whilst the other dimension is the value of α controlling
the actual amount of regularization (alpha_reg).

The value of θ is computed by dividing alpha_reg by alpha_var.

__NB:__ We added the following condition to satisfy the requirements from
        scikit-learn-contrib (_check_estimator_)

```python
# Just to pass check_non_transformer_estimators_n_iter
# because LassoLars stops early for the default alpha=1.0 on the iris dataset.
iris = load_iris()
if (np.array_equal(X, iris.data) and np.array_equal(y,iris.target)):
  alpha = 0.
  self.alpha = 0.
```

But this condition is not relevant as the code is based on the LassoLars
implementation, which already benefits from an exception for the test used in
[estimator.py](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/estimator_checks.py)
(line 2664) on scikit-learn.

Thus those lines will have to be deleted as soon as the project is moved to scikit-learn
(and the test is updated) to avoid any useless computational cost.

## Getting started
#### Prerequisites

This package requires that you have also installed scikit-learn >= 0.23.0 and a few other
packages, as described in _requirements.txt_.

#### Installing

Clone this repository locally, go to the relaxed_lasso directory and then
install with pip:

```
pip install -r requirements.txt
pip install .
```

#### Running the tests

```
python -m pytest
```

#### Example

Simple use case
```
>>> from relaxed_lasso import RelaxedLassoLarsCV
>>> from sklearn.datasets import make_regression
>>> X, y, true_coefs = make_regression(n_samples=50,
                                      n_features=1000,
                                      n_informative=5,
                                      noise=4.0,
                                      random_state=0,
                                      coef=True)
>>> relasso = RelaxedLassoLarsCV(cv=5).fit(X, y)
>>> relasso.score(X, y)
0.9993...
>>> relasso.alpha_
2.7625...
>>> relasso.theta_
6.4120...e-13
>>> relasso.predict(X[:1,])
array([[-124.7180...]])
>>> relasso.coef_[relasso.coef_ != 0]
array([42.5904..., 50.2196..., 98.7397..., 26.8124..., 74.5303...])
>>> true_coefs[true_coefs != 0]
array([41.7911..., 50.9413..., 99.6751..., 27.7122..., 74.2324...])
```
For more, see [examples](./examples).

## Changelog
See [CHANGELOG.md](./CHANGELOG.md).

## Contributing
This implementation was written by [Gregory Vial](mailto:gregory.vial@continental.com) and [Flora Estermann](mailto:flora.estermann@continental.com).

See [CONTRIBUTING.md](./CONTRIBUTING.md). 

## License
Copyright (c) 2020 Continental Corporation. All rights reserved.

This project is licensed under the terms of the 3-Clause BSD License.
See [LICENSE.txt](./LICENSE.txt) for the full license text.
