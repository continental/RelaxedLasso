import pytest
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLars, LassoLarsCV, LinearRegression
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.model_selection import KFold
from sklearn.datasets import make_regression

from relaxed_lasso import RelaxedLassoLars, RelaxedLassoLarsCV
from relaxed_lasso import relasso_lars_path


# Create highly colinear dataset for regression.
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

lasso = LassoLarsCV(cv=5).fit(X_train, y_train)
alpha = lasso.alpha_

# For testing when X input has a single feature
Xa, ya = make_regression(n_samples=50,
                         n_features=1,
                         random_state=0,
                         coef=False)

# For testing when y output vector is multidimensionnal
Xb, yb = make_regression(n_samples=50,
                         n_features=10,
                         n_informative=3,
                         n_targets=2,
                         noise=2,
                         random_state=0,
                         coef=False)

# For testing when extrapolations do cross zero for some sets
X1 = np.array([[1.41029989e+00, 1.45268968e+00, 5.99769397e-01,
                1.13684110e+00, 3.06489880e-01],
               [-2.46677388e-01, 9.78422980e-01, -2.64202914e-01,
                -6.79972180e-02, 3.79508942e-01],
               [-1.24038440e+00, -2.46787653e-01, 1.13914794e-01,
                -5.83695285e-01, -1.12026312e-01],
               [-1.99505603e-01, -4.71148377e-01, -5.55252507e-01,
                2.28270105e+00, 2.07312643e+00],
               [1.60684757e+00, 1.60979535e+00, 1.70659724e+00,
                -1.72166846e-01, 3.42278052e-01],
               [-5.61663629e-02, -2.61004878e-04, 4.55467197e-01,
                -2.35717905e-01, -5.45925262e-01],
               [1.12859835e+00, -4.18807781e-02, -3.02089375e+00,
                -1.02441435e+00, 1.70315086e-01],
               [-2.46458954e-01, 8.15046392e-01, 3.00594276e-01,
                2.53847641e-01, 2.95552912e-02]])
y1 = np.array([20.26793157, 2.18151214, -9.53258406, -3.86202872, 26.71976978,
               2.80537018, -1.44973078, 3.40853846])

X2 = np.array([[-0.3776794, 0.36456531, -1.01243111, -0.38480894, -0.60915583],
               [0.7870042, 1.38241241, 2.61123082, 1.8878823, 0.39648371],
               [0.52548127, 0.14017289, 0.79806355, 0.41964711, 0.16745396],
               [0.95053531, -0.5085372, -0.16109607, 0.01948371, -0.93249031],
               [0.77518773, 0.5312382, 0.07886306, 0.12283602, -0.66082482],
               [-0.72813159, -1.71027536, -0.99093288, -2.04303984,
                1.78732814],
               [-0.04580302, 0.3174949, 0.82426606, 0.87401556, -0.46187965],
               [0.01989191, -0.83380189, 0.34284543, 1.00724264, 1.44058752]])
y2 = np.array([-4.04440993, 20.14707046, 7.55346146, 4.38411891,
               8.12528819, -16.48548726, 3.02433457, -3.16326688])


@pytest.mark.parametrize("fit_path", [True, False])
def test_theta_equal_1(fit_path):
    # Validate that Relaxed Lasso with theta=1 is equivalent to Lasso.
    relasso = RelaxedLassoLars(alpha, 1, fit_path=fit_path).fit(X_train,
                                                                y_train)
    lasso_prediction = lasso.predict(X_test)
    relasso_prediction = relasso.predict(X_test)
    assert_array_almost_equal(lasso_prediction, relasso_prediction)


def test_theta_equal_0():
    # Validate that Relaxed Lasso with theta=0 is equivalent to OLS.
    relasso = RelaxedLassoLars(alpha, 0, fit_path=True).fit(X_train,
                                                                y_train)
    mask = relasso.active_
    lr = LinearRegression(normalize=True).fit(X_train[:, mask], y_train)
    ols_prediction = lr.predict(X_test[:, mask])
    relasso_prediction = relasso.predict(X_test)
    assert_array_almost_equal(ols_prediction, relasso_prediction)


@pytest.mark.parametrize("fit_path", [True, False])
@pytest.mark.parametrize("theta", [1.0, 0.5, .1])
def test_simple_vs_refined_algorithm(theta, fit_path):
    # Test the consistency of the results between the 2 versions of
    # the algorithm.

    # Simple Algorithm (2 steps of Lasso Lars)
    lasso1 = LassoLars(alpha=alpha)
    lasso1.fit(X_train, y_train)
    X1 = X_train.copy()
    X1[:, lasso1.coef_ == 0] = 0

    lasso2 = LassoLars(alpha=alpha*theta)
    lasso2.fit(X1, y_train)
    pred_simple = lasso2.predict(X_test)

    # Refined Algorithm
    relasso = RelaxedLassoLars(alpha=alpha, theta=theta, fit_path=fit_path)
    relasso.fit(X_train, y_train)
    pred_refined = relasso.predict(X_test)

    assert_array_almost_equal(pred_simple, pred_refined)
    assert_array_almost_equal(lasso2.coef_, relasso.coef_)
    assert_almost_equal(lasso2.score(X_test, y_test),
                        relasso.score(X_test, y_test),
                        decimal=2)


def test_relaxed_lasso_lars():
    # Relaxed Lasso regression convergence test using score.

    # With more samples than features
    X1, y1 = make_regression(n_samples=50,
                             n_features=10,
                             n_informative=3,
                             noise=2,
                             random_state=0,
                             coef=False)

    relasso = RelaxedLassoLars()
    relasso.fit(X1, y1)

    assert relasso.coef_.shape == (X1.shape[1],)
    assert relasso.score(X1, y1) > 0.9

    # With more features than samples
    X2, y2 = make_regression(n_samples=50,
                             n_features=100,
                             n_informative=3,
                             noise=2,
                             random_state=0,
                             coef=False)

    relasso = RelaxedLassoLars()
    relasso.fit(X2, y2)

    assert relasso.coef_.shape == (X2.shape[1], )
    assert relasso.score(X2, y2) > 0.9


@pytest.mark.parametrize("theta", [1, .5])
@pytest.mark.parametrize("X, y", [(X, y), (Xa, ya), (Xb, yb)])
def test_shapes(X, y, theta):
    # Test shape of attributes.
    alpha = .5
    relasso = RelaxedLassoLars(alpha, theta)
    relasso.fit(X, y)

    # Multi-targets
    if type(y[0]) == np.ndarray:
        n_alphas = len(relasso.alphas_[0])
        assert len(relasso.alphas_) == y.shape[1]
        assert relasso.alphas_[0].shape == (n_alphas,)
        assert relasso.coef_.shape == (y.shape[1], X.shape[1])
        assert len(relasso.coef_path_) == y.shape[1]
        if len(relasso.alphas_[0]) > 1:
            if theta == 1:
                assert relasso.coef_path_[0].shape == (X.shape[1],
                                                       n_alphas,
                                                       n_alphas - 1)
            else:
                assert relasso.coef_path_[0].shape[1] > \
                       relasso.coef_path_[0].shape[2]
        else:
            assert relasso.coef_path_[0].shape == (X.shape[1], 1, 1)
        assert relasso.intercept_.shape == (y.shape[1],)

    # 1-target
    else:
        print(relasso.alphas_)
        n_alphas = len(relasso.alphas_)
        assert relasso.alphas_.shape == (n_alphas,)
        assert relasso.coef_.shape == (X.shape[1],)
        if len(relasso.alphas_) > 1:
            if theta == 1:
                assert relasso.coef_path_.shape == (X.shape[1],
                                                    n_alphas,
                                                    n_alphas - 1)
            else:
                assert relasso.coef_path_.shape[1] > \
                       relasso.coef_path_.shape[2]
        else:
            assert relasso.coef_path_.shape == (X.shape[1], 1, 1)


@pytest.mark.parametrize("X, y", [(X, y), (Xa, ya)])
def test_relaxed_lasso_lars_cv(X, y):
    # Idem for RelaxedLassoLarsCV
    relasso_cv = RelaxedLassoLarsCV()
    relasso_cv.fit(X, y)
    assert relasso_cv.coef_.shape == (X.shape[1],)
    assert type(relasso_cv.intercept_) == np.float64

    cv = KFold(5)
    relasso_cv.set_params(cv=cv)
    relasso_cv.fit(X, y)
    assert relasso_cv.coef_.shape == (X.shape[1],)
    assert type(relasso_cv.intercept_) == np.float64


# def test_x_none_gram_none_raises_value_error():
    # Test that relasso_lars_path with no X and Gram raises exception.
#    Xy = np.dot(X.T, y)
#    assert_raises(ValueError, relasso_lars_path, None, y, Gram=None, Xy=Xy)


def test_no_path():
    # Test that the 'return_path=False' option returns the correct output.
    alphas_, _, coef_path_ = relasso_lars_path(X, y)
    alpha_, _, coef = relasso_lars_path(X, y, return_path=False)

    # coef_path : array, shape (n_features, n_alphas + 1, n_alphas)
    assert_array_almost_equal(coef, coef_path_[:, -1, -1])
    assert alpha_ == alphas_[-1]


def test_no_path_precomputed():
    # Test that the 'return_path=False' option with Gram remains correct.
    G = np.dot(X.T, X)
    alphas_, _, coef_path_ = relasso_lars_path(
        X, y, method='lasso', Gram=G)
    alpha_, _, coef = relasso_lars_path(
        X, y, method='lasso', Gram=G, return_path=False)

    assert_array_almost_equal(coef, coef_path_[:, -1, -1])
    assert alpha_ == alphas_[-1]


def test_no_path_all_precomputed():
    # Test that the 'return_path=False' option with Gram and Xy
    # remains correct.
    G = np.dot(X.T, X)
    Xy = np.dot(X.T, y)

    alphas_, _, coef_path_ = relasso_lars_path(
        X, y, method='lasso', Xy=Xy, Gram=G, alpha_min=0.9)
    alpha_, _, coef = relasso_lars_path(
        X, y, method='lasso', Gram=G, Xy=Xy, alpha_min=0.9, return_path=False)

    assert_array_almost_equal(coef, coef_path_[:, -1, -1])

    assert alpha_ == alphas_[-1]


def test_relasso_lars_path_length():
    # Test that the path length of the RelaxedLassoLars is right.

    relasso = RelaxedLassoLars(alpha=0.2)
    relasso.fit(X, y)
    relasso2 = RelaxedLassoLars(alpha=relasso.alphas_[2])
    relasso2.fit(X, y)

    assert_array_almost_equal(relasso.alphas_[:3], relasso2.alphas_)

    # Also check that the sequence of alphas is always decreasing
    assert np.all(np.diff(relasso.alphas_) < 0)


def test_singular_matrix():
    # Test when input is a singular matrix.
    X1 = np.array([[1, 1.], [1., 1.]])
    y1 = np.array([1, 1])
    _, _, coef_path = relasso_lars_path(X1, y1)

    assert_array_almost_equal(coef_path.T[-1, :], [[0, 0], [1, 0]])


# Removing for now as not relevant with current version
# @pytest.mark.parametrize("X, y", [(X, y), (X1, y1), (X2, y2)])
# def test_coefficient_sign_change(X, y):
#    # Test that extrapolations does not cross zero
#    alphas_, _, coefs_ = relasso_lars_path(X, y, method='lasso')
#    tol = np.finfo(np.float).eps
#    for i in range(coefs_.shape[2]):
#        for j in range(i, coefs_.shape[1]-1):
#            coefs = coefs_[:, j:j+2, i]
#            coefs[abs(coefs) < tol] = 0
#            assert np.all(np.prod(np.sign(coefs), axis=1) >= 0)


@pytest.mark.parametrize("X, y", [(X, y), (Xa, ya), (Xb, yb)])
def test_no_fit_path(X, y):
    # Test that the 'fit_path=False' option return the correct attributes.
    relasso = RelaxedLassoLars(fit_path=False)
    relasso.fit(X, y)
    try:
        print(relasso.coef_path_)
    except AttributeError:
        error = 1  # coef_path doesn't exist

    assert error == 1


@pytest.mark.parametrize("X, y", [(X, y), (Xa, ya), (Xb, yb)])
def test_coef_path_scaling(X, y):
    """Test that coef_ and coef_path_ are both normalized by X_scale"""
    relasso = RelaxedLassoLars(fit_path=True).fit(X, y)

    # Multi-targets
    if type(y[0]) == np.ndarray:
        print('Multi')
        for i in range(y.shape[1]):
            assert_array_equal(relasso.coef_[i],
                               relasso.coef_path_[i][:, -1, -1])
    # 1-target
    else:
        assert_array_equal(relasso.coef_, relasso.coef_path_[:, -1, -1])
