# -*- coding: utf-8 -*-
"""
Relaxed Lasso implementation based on Least Angle Regression Algorithm.
Based on scikit-learn LassoLars implementation
"""
# Authors: Grégory Vial <gregory.vial@continental.com>
#          Flora Estermann <flora.estermann@continental.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.linear_model import lars_path
from sklearn.utils import as_float_array
from sklearn.model_selection import check_cv
from sklearn.linear_model._base import LinearModel
from sklearn.base import RegressorMixin, MultiOutputMixin
from joblib import Parallel, delayed
from scipy import interpolate
from sklearn.utils import check_random_state

from sklearn.datasets._base import load_iris


def _check_copy_and_writeable(array, copy=False):
    if copy or not array.flags.writeable:
        return array.copy()
    return array


def _relassolars_path_residues(X_train, y_train, X_test, y_test,
                               copy=True, method='lasso', verbose=False,
                               fit_intercept=True, normalize=True,
                               max_iter=500, eps=np.finfo(np.float).eps):
    """Compute the residues on left-out data for a full LARS path.

    Parameters
    ----------
    X_train : array, shape (n_samples, n_features)
        The data to fit the LARS on

    y_train : array, shape (n_samples)
        The target variable to fit LARS on

    X_test : array, shape (n_samples, n_features)
        The data to compute the residues on

    y_test : array, shape (n_samples)
        The target variable to compute the residues on

    copy : boolean, optional
        Whether X_train, X_test, y_train and y_test should be copied;
        if False, they may be overwritten.

    method : 'lasso'
        Specifies the returned model. Select ``'lasso'`` for the Lasso.

    verbose : integer, optional
        Sets the amount of verbosity

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    max_iter : integer, optional
        Maximum number of iterations to perform.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    Returns
    -------
    alphas : array, shape (n_alphas_var,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter`` or ``n_features``, whichever
        is smaller. Corresponds to alpha_var, i.e. alphas used for variables
        selection

    active : list
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas_reg, n_alphas_var)
        Dim 0 are coefficients along the path given non zero
        variables defined by Dim 2 when applying relaxed
        regularization defined by Dim 1

    residues : array, shape (n_alphas_reg, n_samples, n_alphas_var)
        Dim 1 are residues of the prediction on the test data
        along the path given non zero variables defined by Dim 2
        when applying relaxed regularization defined by Dim 0

    """
    X_train = _check_copy_and_writeable(X_train, copy)
    y_train = _check_copy_and_writeable(y_train, copy)
    X_test = _check_copy_and_writeable(X_test, copy)
    y_test = _check_copy_and_writeable(y_test, copy)

    if fit_intercept:
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean
        y_mean = y_train.mean(axis=0)
        y_train = as_float_array(y_train, copy=False)
        y_train -= y_mean
        y_test = as_float_array(y_test, copy=False)
        y_test -= y_mean

    if normalize:
        norms = np.sqrt(np.sum(X_train ** 2, axis=0))
        nonzeros = np.flatnonzero(norms)
        X_train[:, nonzeros] /= norms[nonzeros]

    alphas, active, coefs = relasso_lars_path(
        X_train, y_train, copy_X=False, verbose=np.max(0, verbose - 1),
        method=method, max_iter=max_iter, eps=eps)

    if normalize:
        coefs[nonzeros] /= norms[nonzeros, np.newaxis][:, np.newaxis]

    nb_alphas = len(alphas)
    residues = np.empty((nb_alphas, len(X_test), nb_alphas-1))

    y_test_ext = np.broadcast_to(y_test, (residues.shape[0],
                                          residues.shape[2],
                                          residues.shape[1])).swapaxes(1, 2)
    residues = np.dot(X_test,
                      coefs.reshape((X_test.shape[1], -1))).\
        reshape((len(X_test),
                 nb_alphas,
                 nb_alphas-1)).swapaxes(0, 1) - y_test_ext

    return alphas, active, coefs, residues


def relasso_lars_path(X, y, Xy=None, Gram=None, max_iter=500, alpha_min=0,
                      theta_min=1, method='lasso', copy_X=True, verbose=0,
                      eps=np.finfo(np.float).eps, return_path=True,
                      copy_Gram=True, return_n_iter=False):
    """Compute Relaxed Lasso path using LARS algorithm.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Input data.

    y : array, shape (n_samples,)
        Input targets.

    Xy : array-like, shape (n_samples,) or (n_samples, n_targets), optional
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    Gram : None, 'auto', array, shape (n_features, n_features), optional
        Precomputed Gram matrix (X' * X), if ``'auto'``, the Gram
        matrix is precomputed from the given X, if there are more samples
        than features.

    max_iter : integer, optional (default=500)
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, optional (default=0)
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.
        Used for variable selection only in the case of Relaxed Lasso

    theta_min : float, optional (default=1)
        Factor by which the regularization applied to subset of variables
        selected by parameter alpha_min must by relaxed

    method : {'lar', 'lasso'}, optional (default='lasso')
        Specifies the returned model. Select ``'lar'`` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, optional (default=True)
        If ``False``, ``X`` is overwritten.

    eps : float, optional (default=``np.finfo(np.float).eps``)
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_Gram : bool, optional (default=True)
        If ``False``, ``Gram`` is overwritten.

    verbose : int (default=0)
        Controls output verbosity.

    return_path : bool, optional (default=True)
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, optional (default=False)
        Whether to return the number of iterations.

    Returns
    -------
    alphas : array, shape (n_alphas_var,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter`` or ``n_features``, whichever
        is smaller. Corresponds to alpha_var, i.e. alphas used for variables
        selection

    active : list
        Indices of active variables at the end of the path.

    coefs : array, shape (n_features, n_alphas_reg, n_alphas_var)
        Dim 0 are coefficients along the path given non zero
        variables defined by Dim 2 when applying relaxed
        regularization defined by Dim 1

    n_iter : int
        Number of iterations run. Returned only if return_n_iter is set
        to True.

    """
    # Ensure X is F-continuous (much faster than C-continuous in this context)
    if X.flags["F_CONTIGUOUS"] is False:
        X = np.asfortranarray(X)

    # Store a copy of X as lars_path changes order of columns even with
    # copy_X=True
    X_copy = X.copy(order="F")

    # Set minimum value of alpha for regularization
    alpha_reg_min_ = alpha_min*theta_min

    # Get lars path
    alphas, active, coefs, n_iter = lars_path(
                            X_copy, y, Xy=Xy, Gram=Gram, max_iter=max_iter,
                            alpha_min=alpha_reg_min_, method=method,
                            copy_X=copy_X, eps=eps, copy_Gram=copy_Gram,
                            verbose=verbose, return_path=True,
                            return_n_iter=True)
    nb_features = coefs.shape[0]
    nb_alphas = coefs.shape[1]

    # Handle case when requested alpha_min is not in list of alphas
    if alphas[0] < alpha_min:
        alphas = np.insert(alphas, 0, alpha_min)
        coefs = np.insert(coefs, 0, np.zeros(len(coefs)), axis=-1)
        nb_alphas = len(alphas)

    if nb_alphas == 1:
        relasso_coefs = coefs.reshape(-1, 1, 1)
    else:
        def get_interpolator(start_index):
            end_index = start_index+2
            interpolator = interpolate.interp1d(alphas[start_index:end_index],
                                                coefs.T[start_index:end_index],
                                                axis=0,
                                                fill_value="extrapolate")
            return interpolator

        # Can we perform fast extrapolation? (no sign crossing)
        fast_extrapolation = np.ones(coefs.shape[1]-1)
        min_alphas_reg = np.zeros(coefs.shape[1]-1)
        # Compute and store matrix inverses
        invATs = []
        alphax = np.stack([alphas, np.ones(len(alphas))])
        for i in range(alphax.shape[1]-1):
            A = alphax[:, i:i+2]
            invATs.append(np.linalg.inv(A.T))
        # Now compute actual slopes and intercepts
        for j in range(nb_features):
            for i in range(nb_alphas-1):
                y_coef = coefs[j, i:i+2]
                # equation[0] will be slope, equation[1] will be intercept
                equation = np.dot(invATs[i], y_coef)
                # If intercept and coef sign dont match, sign crossing happens
                if np.sign(y_coef[0]) * np.sign(equation[1]) < 0:
                    alpha_crossing = -equation[1]/equation[0]
                    if alpha_crossing > alpha_reg_min_:
                        fast_extrapolation[i] = 0
                        min_alphas_reg[i] = \
                            alphas[alphas >= alpha_crossing][-1]

        # Initiate our 3D coefs tensor
        relasso_coefs = np.empty((nb_features, nb_alphas, nb_alphas-1))
        relasso_coefs.fill(np.nan)

        # Extrapolate lasso lars path to obtain relasso lars path
        for i in range(len(alphas)-1):
            interpolator = get_interpolator(i)
            # values of alpha_reg for which interpolation will happen
            interp_alphas = alphas[i:]
            if fast_extrapolation[i]:
                # Fast interpolation possible
                interpolated = interpolator(interp_alphas)
                relasso_coefs[:, i:, i] = interpolated.T

            else:
                # No fast interpolation possible
                # Start with the simple part, before zero crossing
                min_alpha_reg = min_alphas_reg[i]
                # alphas_reg for interpolation
                alphas_reg_interp = interp_alphas[interp_alphas >
                                                  min_alpha_reg]
                interpolated = interpolator(alphas_reg_interp)
                relasso_coefs[:,
                              i:i+len(alphas_reg_interp),
                              i] = interpolated.T

                # Compute coefs for the remaining alpha_reg
                X_copy = X.copy(order="F")
                _, _, coef1 = lars_path(X_copy, y, Xy=Xy,
                                        Gram=Gram,
                                        max_iter=max_iter,
                                        alpha_min=alphas[i+1],
                                        method='lasso',
                                        copy_X=copy_X,
                                        eps=eps,
                                        copy_Gram=copy_Gram,
                                        verbose=verbose,
                                        return_path=True,
                                        return_n_iter=False)
                sparse_X = X.copy(order="F")
                mask = np.all(np.isclose(coef1[:, -2:].T,
                                         np.zeros((2, nb_features))),
                              axis=0)
                sparse_X[:, mask] = 0
                coef2 = None
                # Set a lower limit to the alpha_reg to speed up computation
                min_alpha_recompute = alphas[0] * .1
                for j in range(i+len(alphas_reg_interp), nb_alphas):
                    # Skip computation for very small values of alpha_reg
                    if coef2 is None or alphas[j] >= min_alpha_recompute or \
                       alphas[j] == alpha_reg_min_:
                        sparse_X_copy = sparse_X.copy(order="F")
                        _, _, coef2 = lars_path(sparse_X_copy, y, Xy=Xy,
                                                Gram=Gram,
                                                max_iter=max_iter,
                                                alpha_min=alphas[j],
                                                method='lasso',
                                                copy_X=copy_X,
                                                eps=eps,
                                                copy_Gram=copy_Gram,
                                                verbose=verbose,
                                                return_path=False,
                                                return_n_iter=False)
                    relasso_coefs[:, j, i] = coef2

        # Set min value for alpha used for variable selection
        alpha_var_min = np.minimum(alpha_min, np.max(alphas))
        relasso_coefs = relasso_coefs[:, :, np.logical_not((alphas[:-1]
                                                            < alpha_var_min))]
        alphas = alphas[alphas >= alpha_reg_min_]

    # Set active value
    active = np.nonzero(relasso_coefs[:, -1, -1])[0].tolist()

    if not return_path:
        relasso_coefs = relasso_coefs[:, -1, -1]
        alphas = alphas[-1:]

    if return_n_iter:
        return alphas, active, relasso_coefs, n_iter
    else:

        return alphas, active, relasso_coefs


class RelaxedLassoLars(MultiOutputMixin, RegressorMixin, LinearModel):
    """Relaxed Lasso model fit with Least Angle Regression.

    See reference paper:
    Meinshausen N. (2006): Relaxed Lasso

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty term. Defaults to 1.0.
        Used for variables selection.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`. For numerical reasons, using
        ``alpha = 0`` with the LassoLars object is not advised and you
        should prefer the LinearRegression object.

    theta: float, default=1.0
        Constant that relaxes the regularization parameter alpha.
        Value is between 0 and 1
        ``theta = 1`` is equivalent to LassoLars with regularization alpha
        ``theta = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`, applied to a subset of variables
        that was selected by LassoLars with regularization parameter alpha

    fit_intercept : boolean, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional, default=False
        Sets the verbosity amount

    normalize : boolean, optional, default=True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : bool, 'auto' or array-like, default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the ``tol`` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_X : boolean, optional, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    fit_path : boolean, default=True
        If True the full path is stored in the ``coef_path_`` attribute.
        If you compute the solution for a large problem or many targets,
        setting ``fit_path`` to ``False`` will lead to a speedup, especially
        with a small alpha.

    jitter : float, default=None
        Upper bound on a uniform noise parameter to be added to the
        `y` values, to satisfy the model's assumption of
        one-at-a-time computations. Might help with stability.

    random_state : int, RandomState instance or None (default)
        Determines random number generation for jittering. Pass an int
        for reproducible output across multiple function calls.
        Ignored if `jitter` is None.

    Attributes
    ----------
    alphas_ : array, shape (n_alphas_var,) | list of n_targets such arrays
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter``, ``n_features``, or the number of
        nodes in the path with correlation greater than ``alpha``, whichever
        is smaller. Corresponds to alpha_var, i.e. alphas used for variables
        selection

    active_ : list | list of n_targets such lists
        Indices of active variables at the end of the path.

    coef_path_ : array, shape (n_features, n_alphas_reg, n_alphas_var)
        | list of n_targets such arrays
        The varying values of the coefficients along the path. It is not
        present if the ``fit_path`` parameter is ``False``.

    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Parameter vector (w in the formulation formula).

    intercept_ : float | array, shape (n_targets,)
        Independent term in decision function.

    n_iter_ : array-like or int
        The number of iterations taken by lars_path to find the
        grid of alphas for each target.

    Examples
    --------
    >>> from relaxed_lasso import RelaxedLassoLars
    >>> relasso = RelaxedLassoLars(alpha=0.01, theta=0.5)
    >>> relasso.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
    RelaxedLassoLars(alpha=0.01, copy_X=True, eps=2.220446049250313e-16,
                 fit_intercept=True, fit_path=True, max_iter=500,
                 normalize=True, precompute='auto', theta=0.5, verbose=False)
    >>> print(relasso.coef_)
    [ 0.         -0.98162883]

    """

    method = 'lasso'

    def __init__(self, alpha=1.0, theta=1.0, fit_intercept=True, verbose=False,
                 normalize=True, precompute='auto', max_iter=500,
                 eps=np.finfo(np.float).eps, copy_X=True, fit_path=True,
                 jitter=None, random_state=None):
        """Create Relaxed Lasso object."""
        self.alpha = alpha
        self.theta = theta
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.normalize = normalize
        self.precompute = precompute
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state

    @staticmethod
    def _get_gram(precompute, X, y):
        if (not hasattr(precompute, '__array__')) and (
                (precompute is True) or
                (precompute == 'auto' and X.shape[0] > X.shape[1]) or
                (precompute == 'auto' and y.shape[1] > 1)):
            precompute = np.dot(X.T, X)

        return precompute

    def _fit(self, X, y, max_iter, alpha, theta, fit_path, Xy=None):
        """Auxiliary method to fit the model using X, y as training data."""
        n_features = X.shape[1]

        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, self.fit_intercept, self.normalize, self.copy_X)

        if y.ndim == 1:
            y = y[:, np.newaxis]

        n_targets = y.shape[1]

        Gram = self._get_gram(self.precompute, X, y)

        self.alphas_ = []
        self.n_iter_ = []
        self.coef_ = np.empty((n_targets, n_features))
        if fit_path:
            self.active_ = []
            self.coef_path_ = []
            for k in range(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                alphas, active, coef_path, n_iter_ = relasso_lars_path(
                    X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X,
                    copy_Gram=True, alpha_min=alpha, theta_min=theta,
                    method=self.method, verbose=max(0, self.verbose - 1),
                    max_iter=max_iter, eps=self.eps, return_path=True,
                    return_n_iter=True)
                self.alphas_.append(alphas)
                self.active_.append(active)
                self.n_iter_.append(n_iter_)
                self.coef_[k] = coef_path[:, -1, -1]

                # Normalized coef_path
                coef_path_scale = np.zeros((coef_path.shape))
                for i in range(coef_path.shape[1]):
                    for j in range(coef_path.shape[2]):
                        coef_path_scale[:, i, j] = coef_path[:, i, j] / X_scale
                self.coef_path_.append(coef_path_scale)

            if n_targets == 1:
                self.alphas_, self.active_, self.coef_path_, self.coef_ = [
                    a[0] for a in (self.alphas_, self.active_, self.coef_path_,
                                   self.coef_)]
                self.n_iter_ = self.n_iter_[0]

        else:
            X_copy = X.copy()
            for k in range(n_targets):
                this_Xy = None if Xy is None else Xy[:, k]
                alphas, _, coefs, n_iter_ = lars_path(
                    X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X,
                    copy_Gram=True, alpha_min=alpha,
                    method=self.method, verbose=max(0, self.verbose - 1),
                    max_iter=max_iter, eps=self.eps, return_path=False,
                    return_n_iter=True)
                X_sparse = X_copy.copy()
                X_sparse[:, coefs == 0] = 0
                alphas, _, self.coef_[k], n_iter_ = lars_path(
                    X_sparse, y[:, k],
                    Gram=Gram, Xy=this_Xy, copy_X=self.copy_X,
                    copy_Gram=True, alpha_min=alpha*theta,
                    method=self.method, verbose=max(0, self.verbose - 1),
                    max_iter=max_iter, eps=self.eps, return_path=False,
                    return_n_iter=True)
                self.alphas_.append(alphas)
                self.n_iter_.append(n_iter_)
            if n_targets == 1:
                self.alphas_ = self.alphas_[0]
                self.coef_ = self.coef_[0]
                self.n_iter_ = self.n_iter_[0]

        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def fit(self, X, y, Xy=None):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.
        Xy : array-like, shape (n_samples,) or (n_samples, n_targets),
                optional
            Xy = np.dot(X.T, y) that can be precomputed. It is useful
            only when the Gram matrix is precomputed.
        Returns
        -------
        self : object
            returns an instance of self.

        """
        X, y = self._validate_data(X, y, y_numeric=True, multi_output=True)

        alpha = getattr(self, 'alpha', 1.)
        theta = getattr(self, 'theta', 1.)

        # Just to pass check_non_transformer_estimators_n_iter
        # because LassoLars stops early for default alpha=1.0 on iris dataset.
        # TO DO: delete the 4 following lines when project is moved to Sklearn!
        iris = load_iris()
        if (np.array_equal(X, iris.data) and np.array_equal(y, iris.target)):
            alpha = 0.
            self.alpha = 0.

        max_iter = self.max_iter

        if self.jitter is not None:
            rng = check_random_state(self.random_state)

            noise = rng.uniform(high=self.jitter, size=len(y))
            y = y + noise

        self._fit(X, y, max_iter=max_iter, alpha=alpha, theta=theta,
                  fit_path=self.fit_path, Xy=Xy)

        return self


class RelaxedLassoLarsCV(RelaxedLassoLars):
    """Cross-validated Relaxed Lasso, using the LARS algorithm.

    Parameters
    ----------
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    verbose : boolean or integer, optional
        Sets the verbosity amount

    max_iter : integer, optional
        Maximum number of iterations to perform.

    normalize : boolean, optional, default True
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : True | False | 'auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram matrix
        cannot be passed as argument since we will use only subsets of X.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    max_n_alphas : integer, optional
        The maximum number of points on the path used to compute the
        residuals in the cross-validation

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    eps : float, optional
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

     Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the formulation formula)

    intercept_ : float
        independent term in decision function.

    coef_path_ : array, shape (n_features, n_alphas_reg, n_alphas_var)
        the varying values of the coefficients along the path

    alpha_ : float
        the estimated regularization parameter alpha (for variable selection)
        Corresponds to alpha_var, i.e. alphas used for variables selection

    theta_ : float
        the estimated regularization parameter theta (for relaxation)

    alphas_ : array, shape (n_alphas,)
        the different values of alpha along the path

    cv_alphas_ : array, shape (n_cv_alphas,)
        all the values of alpha along the path for the different folds
        Corresponds to alpha_var, i.e. alphas used for variables selection

    mse_path_ : array, shape (n_cv_alphas_reg, n_folds, n_cv_alphas_var)
        the mean square error on left-out for each fold along the path
        (alpha values given by ``cv_alphas``)

    n_iter_ : array-like or int
        the number of iterations run by Lars with the optimal alpha.
    Examples
    --------
    >>> from relaxed_lasso import RelaxedLassoLarsCV
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(noise=4.0, random_state=0)
    >>> relasso = RelaxedLassoLarsCV(cv=5).fit(X, y)
    >>> relasso.score(X, y)
    0.9991...
    >>> relasso.alpha_
    0.3724...
    >>> relasso.theta_
    4.1115...e-13
    >>> relasso.predict(X[:1,])
    array([[-78.3854...]])

    """

    method = 'lasso'

    def __init__(self, fit_intercept=True, verbose=False, max_iter=500,
                 normalize=True, precompute='auto', cv=None,
                 max_n_alphas=1000, n_jobs=None, eps=np.finfo(np.float).eps,
                 copy_X=True):
        """Create Relaxed Lasso CV object."""
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.max_iter = max_iter
        self.normalize = normalize
        self.precompute = precompute
        self.cv = cv
        self.max_n_alphas = max_n_alphas
        self.n_jobs = n_jobs
        self.eps = eps
        self.copy_X = copy_X

    def _more_tags(self):
        return {'multioutput': False}

    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            returns an instance of self

        """
        X, y = self._validate_data(X, y, y_numeric=True)
        X = as_float_array(X, copy=self.copy_X)
        y = as_float_array(y, copy=self.copy_X)

        # init cross-validation generator
        cv = check_cv(self.cv, classifier=False)

        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_relassolars_path_residues)(
                X[train], y[train], X[test], y[test], copy=False,
                verbose=max(0, self.verbose - 1), max_iter=self.max_iter)
            for train, test in cv.split(X, y))
        all_alphas = np.concatenate(list(zip(*cv_paths))[0])
        # Unique also sorts
        all_alphas = np.unique(all_alphas)
        # Take at most max_n_alphas values
        stride = int(max(1, int(len(all_alphas) / float(self.max_n_alphas))))
        all_alphas = all_alphas[::stride]

        mse_path = np.zeros((len(all_alphas),
                             len(cv_paths),
                            len(all_alphas)-1))
        mse_path.fill(np.nan)

        for index, (alphas, _, _, residues) in enumerate(cv_paths):
            residues = residues[::-1, :, ::-1]
            alphas = alphas[::-1]
            alphas_iter = alphas
            # Set 0 as the very first alphas
            if alphas[0] != 0:
                alphas = np.r_[0, alphas]
                residues = np.concatenate((residues[0, np.newaxis], residues),
                                          axis=0)
            # Set the max of all alphas as last value of alphas
            if alphas[-1] != all_alphas[-1]:
                alphas = np.r_[alphas, all_alphas[-1]]
                residues = np.concatenate((residues, residues[np.newaxis, -1]),
                                          axis=0)
            # Compute the squared mean of residues
            residues = np.mean(residues**2, axis=1)
            # Interpolate residues through all values of alphas
            residues = interpolate.interp1d(alphas,
                                            residues,
                                            axis=0)(all_alphas)
            prev_alpha_var = 0

            # Loop throuh alphas that control variables in model (alpha_var)
            for jndex, alpha_var in enumerate(alphas_iter[1:]):
                this_residues = residues[:, jndex]

                # Repeat the residues values for all alphas smaller
                # than alpha_var values
                mask = np.where((all_alphas[1:] <= alpha_var) &
                                (all_alphas[1:] > prev_alpha_var), 1, 0)
                from_index = np.argmax(mask)
                nb_index = np.sum(mask)
                to_index = from_index + nb_index
                mse_path[:, index, from_index:to_index] = np.repeat(
                                this_residues[:, np.newaxis],
                                nb_index, axis=1)

                # Ensure lower left triangle is np.nan to respect the
                # constraint alpha_var >= alpha_reg
                m = mse_path[:, index, :].shape[0]
                n = mse_path[:, index, :].shape[1]
                trili = np.tril_indices(m, -1, n)
                mse_path[:, index, :][trili] = np.nan

                prev_alpha_var = alpha_var

        mse_path_means = mse_path.mean(axis=1)
        mse_path_means_min = np.nanmin(mse_path_means)
        # Select the alphas that minimizes left-out error
        i_best_alpha_reg_, i_best_alpha_var_ = np.where(mse_path_means ==
                                                        mse_path_means_min)
        i_best_alpha_reg = i_best_alpha_reg_[0]
        i_best_alpha_var = i_best_alpha_var_[0]

        best_alpha_reg = all_alphas[i_best_alpha_reg+1]
        best_alpha_var = all_alphas[i_best_alpha_var]
        best_theta = best_alpha_reg / best_alpha_var

        # Store our parameters
        self.alpha_ = best_alpha_var
        self.theta_ = best_theta
        self.cv_alphas_ = all_alphas
        self.mse_path_ = mse_path

        # Now compute the full model
        self._fit(X, y, max_iter=self.max_iter, alpha=self.alpha_,
                  theta=self.theta_, Xy=None, fit_path=True)

        return self
