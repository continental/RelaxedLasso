# -*- coding: utf-8 -*-
"""Relaxed Lasso implementation based on Coordinate Descent Algorithm.

Based on scikit-learn Lasso implementation
"""
# Authors: Grégory Vial <gregory.vial@continental.com>
#          Flora Estermann <flora.estermann@continental.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse
from sklearn.linear_model import ElasticNet, lasso_path
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.linear_model._base import _pre_fit, _preprocess_data
from sklearn.utils import as_float_array, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.model_selection import check_cv
from joblib import Parallel, delayed
from scipy import interpolate
from sklearn.utils.validation import _deprecate_positional_args


def _check_copy_and_writeable(array, copy=False):
    if copy or not array.flags.writeable:
        return array.copy()
    return array


def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-13, n_alphas=100, normalize=False, copy_X=True):

    """ Compute the grid of alpha values for Relaxed Lasso parameters search

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray of shape (n_samples,)
        Target values

    Xy : array-like of shape (n_features,), default=None
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float, default=1.0
        The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. (currently not
        supported) ``For l1_ratio = 1`` it is an L1 penalty. For
        ``0 < l1_ratio <1``, the penalty is a combination of L1 and L2.

    eps : float, default=1e-13
        Length of the path. ``eps=1e-13`` means that
        ``alpha_min / alpha_max = 1e-13``

    n_alphas : int, default=100
        Number of alphas_var along the regularization path

    fit_intercept : bool, default=True
        Whether to fit an intercept or not

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    if l1_ratio == 0:
        raise ValueError("Automatic alpha grid generation is not supported for"
                         " l1_ratio=0. Please supply a grid by providing "
                         "your estimator with the appropriate `alphas=` "
                         "argument.")
    n_samples = len(y)

    sparse_center = False
    if Xy is None:
        X_sparse = sparse.isspmatrix(X)
        sparse_center = X_sparse and (fit_intercept or normalize)
        X = check_array(X, accept_sparse='csc',
                        copy=(copy_X and fit_intercept and not X_sparse))

        if not X_sparse:
            # X can be touched inplace thanks to the above line
            X, y, _, _, _ = _preprocess_data(X, y, fit_intercept,
                                             normalize, copy=False)
        Xy = safe_sparse_dot(X.T, y, dense_output=True)

        if sparse_center:
            # Workaround to find alpha_max for sparse matrices.
            # since we should not destroy the sparsity of such matrices.
            _, _, X_offset, _, X_scale = _preprocess_data(X, y, fit_intercept,
                                                          normalize,
                                                          return_mean=True)
            mean_dot = X_offset * np.sum(y)

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    if sparse_center:
        if fit_intercept:
            Xy -= mean_dot[:, np.newaxis]
        if normalize:
            Xy /= X_scale[:, np.newaxis]

    # Smallest alpha such that all the coefficients are zero
    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    alphas = np.logspace(np.log10(alpha_max * eps), 1, num=n_alphas)
    alphas[-1] = alpha_max

    return sorted(alphas, reverse=True)


def _relasso_path_residues(X_train, y_train, X_test, y_test,
                           copy=True, verbose=False, max_iter=1000,
                           fit_intercept=True, normalize=False,
                           eps=1e-13, n_alphas=100, alphas=None):
    """Compute the residues on left-out data for a full lasso path.
    Parameters
    ----------
    X_train : array, shape (n_samples, n_features)
        The data to fit the Coordinate Descent on

    y_train : array, shape (n_samples)
        The target variable to fit Coordinate Descent on

    X_test : array, shape (n_samples, n_features)
        The data to compute the residues on

    y_test : array, shape (n_samples)
        The target variable to compute the residues on

    n_alphas : int, default=100
        Number of alphas along the regularization path

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    copy : boolean, optional
        Whether X_train, X_test, y_train and y_test should be copied;
        if False, they may be overwritten.

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

    coefs : array, shape (n_features, n_alphas_reg, n_alphas_var)
        Dim 0 are coefficients along the path given non zero
        variables defined by Dim 2 when applying relaxed
        regularization defined by Dim 1

    dual_gaps : array, shape (n_alphas_var,)
        The dual gaps at the end of the optimization for each alpha.

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

    alphas, coefs, dual_gaps = relasso_path(
        X_train, y_train, copy_X=False, verbose=np.max(0, verbose - 1),
        eps=eps, alphas=alphas)

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

    return alphas, coefs, dual_gaps, residues


def relasso_path(X, y, eps=1e-13, alpha_min=0,
                 theta_min=1, precompute='auto', Xy=None,
                 copy_X=True, coef_init=None, verbose=False,
                 return_n_iter=False, positive=False,
                 return_path=True, n_alphas=100, alphas=None):
    """Compute Relaxed Lasso path with Coordinate Descent.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication. If ``y`` is mono-output then ``X``
        can be sparse.

    y : {array-like, sparse matrix} of shape (n_samples,) or \
        (n_samples, n_outputs)
        Target values

    alpha_min : float, optional (default=0)
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.
        Used for variable selection only in the case of Relaxed Lasso

    theta_min : float, optional (default=1)
        Factor by which the regularization applied to subset of variables
        selected by parameter alpha_min must by relaxed

    eps : float, default=1e-13
        Length of the path. ``eps=1e-13`` means that
        ``alpha_min / alpha_max = 1e-13``

    n_alphas : int, default=100
        Number of alphas along the regularization path

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    precompute : 'auto', bool or array-like of shape (n_features, n_features),\
                 default='auto'
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    Xy : array-like of shape (n_features,) or (n_features, n_outputs),\
         default=None
        Xy = np.dot(X.T, y) that can be precomputed. It is useful
        only when the Gram matrix is precomputed.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    coef_init : ndarray of shape (n_features, ), default=None
        The initial values of the coefficients.

    verbose : bool or int, default=False
        Amount of verbosity.

    return_path : bool, optional (default=True)
        If ``return_path==True`` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        whether to return the number of iterations or not.

    positive : bool, default=False
        If set to True, forces coefficients to be positive.
        (Only allowed when ``y.ndim == 1``).

    Returns
    -------
    alphas : array, shape (n_alphas_var,)
        Maximum of covariances (in absolute value) at each iteration.
        ``n_alphas`` is either ``max_iter`` or ``n_features``, whichever
        is smaller. Corresponds to alpha_var, i.e. alphas used for variables
        selection

    coefs : array, shape (n_features, n_alphas_reg, n_alphas_var)
        Dim 0 are coefficients along the path given non zero
        variables defined by Dim 2 when applying relaxed
        regularization defined by Dim 1

    dual_gaps : array, shape (n_alphas_var,)
        The dual gaps at the end of the optimization for each alpha.

    n_iters : list of int
        The number of iterations taken by the coordinate descent optimizer to
        reach the specified tolerance for each alpha.

    """
    # Ensure X is F-continuous (much faster than C-continuous in this context)
    if X.flags["F_CONTIGUOUS"] is False:
        X = np.asfortranarray(X)

    alpha_reg_min_ = alpha_min * theta_min

    # Get lasso path with coordinate descent
    if return_n_iter:
        alphas, coefs, dual_gaps, n_iter = lasso_path(
                        X, y, eps=eps, n_alphas=n_alphas, alphas=alphas,
                        precompute=precompute, Xy=Xy, copy_X=copy_X,
                        coef_init=coef_init, verbose=verbose,
                        return_n_iter=True, positive=positive)
    else:
        alphas, coefs, dual_gaps = lasso_path(
                        X, y, eps=eps, n_alphas=n_alphas, alphas=alphas,
                        precompute=precompute, Xy=Xy, copy_X=copy_X,
                        coef_init=coef_init, verbose=verbose,
                        return_n_iter=False, positive=positive)

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
                _, coef1, _ = lasso_path(X_copy, y, eps=eps,
                                         n_alphas=None,
                                         alphas=alphas[:i+1],
                                         precompute=precompute,
                                         Xy=Xy, copy_X=copy_X,
                                         coef_init=coef_init,
                                         verbose=verbose,
                                         return_n_iter=False,
                                         positive=positive)
                sparse_X = X.copy(order="F")
                mask = np.all(np.isclose(coef1[:, -2:].T,
                                         np.zeros((2, nb_features))),
                              axis=0)
                sparse_X[:, mask] = 0
                coef2 = None
                # Set a lower limit to the alpha_reg to speed up computation
                min_alpha_recompute = alphas[0] *.1
                for j in range(i+len(alphas_reg_interp), nb_alphas):
                    # Skip computation for very small values of alpha_reg
                    if coef2 is None or alphas[j] >= min_alpha_recompute or \
                       alphas[j] == alpha_reg_min_:
                        sparse_X_copy = sparse_X.copy(order="F")
                        if j == 0:
                            _, coef2, _ = lasso_path(sparse_X_copy, y, eps=eps,
                                                     n_alphas=None,
                                                     alphas=[alphas[j]],
                                                     precompute=precompute,
                                                     Xy=Xy, copy_X=copy_X,
                                                     coef_init=coef_init,
                                                     verbose=verbose,
                                                     return_n_iter=False,
                                                     positive=positive)
                        else:
                            _, coef2, _ = lasso_path(sparse_X_copy, y, eps=eps,
                                                     n_alphas=None,
                                                     alphas=alphas[:j],
                                                     precompute=precompute,
                                                     Xy=Xy, copy_X=copy_X,
                                                     coef_init=coef_init,
                                                     verbose=verbose,
                                                     return_n_iter=False,
                                                     positive=positive)

                    relasso_coefs[:, j, i] = coef2[:, -1]

        # Set min value for alpha used for variable selection
        alpha_var_min = np.minimum(alpha_min, np.max(alphas))
        relasso_coefs = relasso_coefs[:, :, np.logical_not((alphas[:-1]
                                                            < alpha_var_min))]
        alphas = alphas[alphas >= alpha_reg_min_]

    if not return_path:
        relasso_coefs = relasso_coefs[:, -1, -1]
        alphas = alphas[-1:]

    if return_n_iter:
        return alphas, relasso_coefs, dual_gaps, n_iter

    else:
        return alphas, relasso_coefs, dual_gaps


class RelaxedLasso(ElasticNet):
    """Relaxed Lasso model fit with coordinate descent.

    Technically the Lasso model is optimizing the same objective
    function as the Elastic Net with l1_ratio=1.0 (no L2 penalty).

    See reference paper:
    Meinshausen N. (2006): Relaxed Lasso

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    theta: float, default=1.0
        Constant that relaxes the regularization parameter alpha.
        Value is between 0 and 1
        ``theta = 1`` is equivalent to Lasso with regularization alpha
        ``theta = 0`` is equivalent to an ordinary least square, solved
        by :class:`LinearRegression`, applied to a subset of variables
        that was selected by LassoLars with regularization parameter alpha

    eps : float, default=1e-3
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when ``fit_intercept`` is set to False.
        If True, the regressors X will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use
        :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
        on an estimator with ``normalize=False``.

    precompute : 'auto', bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1000
        The maximum number of iterations

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    verbose : bool or int, default=False
        Amount of verbosity.

    Attributes
    ----------
    alphas_ : array, shape (n_alphas_var,) | list of n_targets such arrays
    Maximum of covariances (in absolute value) at each iteration.
    ``n_alphas`` is either ``max_iter``, ``n_features``, or the number of
    nodes in the path with correlation greater than ``alpha``, whichever
    is smaller. Corresponds to alpha_var, i.e. alphas used for variables
    selection

    coefs_ : array, shape (n_features,) or (n_targets, n_features)
        Dim 0 are coefficients along the path given non zero
        variables defined by Dim 2 when applying relaxed
        regularization defined by Dim 1

    dual_gaps_ : array, shape (n_alphas_var,)
        The dual gaps at the end of the optimization for each alpha.

    sparse_coef_ : sparse matrix of shape (n_features, 1) or \
            (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float or ndarray of shape (n_targets,)
        independent term in decision function.

    n_iter_ : int or list of int
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    See also
    --------
    lars_path
    lasso_path
    LassoLars
    LassoCV
    LassoLarsCV
    sklearn.decomposition.sparse_encode

    Notes
    -----
    The algorithm used to fit the model is coordinate descent.

    To avoid unnecessary memory duplication the X argument of the fit method
    should be directly passed as a Fortran-contiguous numpy array.
    """

    @_deprecate_positional_args
    def __init__(self, alpha=1.0, theta=1.0, fit_intercept=True,
                 normalize=False, precompute=False, copy_X=True,
                 max_iter=1000, tol=1e-4, warm_start=False,
                 positive=False, random_state=None, selection='cyclic',
                 eps=1e-3, verbose=False):
        super().__init__(
            alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)
        self.eps = eps
        self.theta = theta
        self.verbose = verbose

    def _fit(self, X, y, max_iter, alpha, theta, X_copied=False):
        """Auxiliary method to fit the model using X, y as training data."""
        n_features = X.shape[1]

        # Ensure copying happens only once, don't do it again if done above.
        # X and y will be rescaled if sample_weight is not None, order='F'
        # ensures that the returned X and y are still F-contiguous.
        should_copy = self.copy_X and not X_copied
        X, y, X_offset, y_offset, X_scale, precompute, Xy = \
            _pre_fit(X, y, None, self.precompute, self.normalize,
                     self.fit_intercept, copy=should_copy)

        X, y = _set_order(X, y, order='F')
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if Xy is not None and Xy.ndim == 1:
            Xy = Xy[:, np.newaxis]

        n_targets = y.shape[1]

        self.alphas_ = []
        self.coef_ = np.empty((n_targets, n_features))
        self.dual_gaps_ = []
        self.n_iter_ = []

        X_copy = X.copy()
        for k in range(n_targets):
            this_Xy = None if Xy is None else Xy[:, k]
            # Model selection
            alphas, coefs, dual_gaps, n_iter = lasso_path(
                 X, y[:, k], eps=self.eps,
                 n_alphas=None, alphas=[alpha],
                 precompute=self.precompute, Xy=this_Xy,
                 copy_X=self.copy_X, coef_init=None,
                 verbose=max(0, self.verbose - 1),
                 return_n_iter=True)
            X_sparse = X_copy.copy()
            X_sparse[:, coefs[:, -1] == 0] = 0

            # Shrinkage estimation
            alphas, coefs, dual_gaps, n_iter = lasso_path(
                 X_sparse, y[:, k], eps=self.eps,
                 n_alphas=None, alphas=[alpha*theta],
                 precompute=self.precompute, Xy=this_Xy,
                 copy_X=self.copy_X, coef_init=None,
                 verbose=max(0, self.verbose - 1),
                 return_n_iter=True)
            self.coef_[k] = coefs[:, -1]
            self.alphas_.append(alphas)
            self.dual_gaps_.append(dual_gaps)
            self.n_iter_.append(n_iter[0])

        if n_targets == 1:
            self.alphas_ = self.alphas_[0]
            self.coef_ = self.coef_[0]
            self.dual_gaps_ = self.dual_gaps_[0]
            self.n_iter_ = self.n_iter_[0]

        self._set_intercept(X_offset, y_offset, X_scale)

        return self

    def fit(self, X, y):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            returns an instance of self.

        """
        X_copied = self.copy_X and self.fit_intercept
        # We expect X and y to be float64 or float32 Fortran ordered
        # arrays when bypassing checks
        X, y = self._validate_data(X, y, accept_sparse='csc',
                                   order='F',
                                   dtype=[np.float64, np.float32],
                                   copy=X_copied, multi_output=True,
                                   y_numeric=True)
        y = check_array(y, order='F', copy=False, dtype=X.dtype.type,
                        ensure_2d=False)

        alpha = getattr(self, 'alpha', 1.)
        theta = getattr(self, 'theta', 1.)

        max_iter = self.max_iter

        self._fit(X, y, max_iter=max_iter, alpha=alpha, theta=theta,
                  X_copied=X_copied)

        return self


class RelaxedLassoCV(RelaxedLasso):
    """Cross-validated Relaxed Lasso, using the Coordinate Descent algorithm.

    Parameters
    ----------
    n_alphas : int, default=100
        Number of alphas to test along the regularization path

    alphas : ndarray, default=None
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    eps : float, default=1e-13
        Length of the path. ``eps=1e-13`` means that
        ``alpha_min / alpha_max = 1e-13``

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

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

     Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the formulation formula)

    dual_gaps_ : array, shape (n_alphas_var,)
        The dual gaps at the end of the optimization for each alpha.

    sparse_coef_ : sparse matrix of shape (n_features, 1)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float
        independent term in decision function.

    alpha_ : float
        the estimated regularization parameter alpha

    theta_ : float
        the estimated regularization parameter theta

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

    """

    def __init__(self, fit_intercept=True, verbose=False,
                 max_iter=1000, normalize=False,
                 precompute='auto', cv=None, tol=1e-4,
                 max_n_alphas=1000, n_jobs=None, eps=1e-13,
                 copy_X=True, positive=False,
                 n_alphas=100, alphas=None):
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
        self.positive = positive
        self.tol = tol
        self.n_alphas = n_alphas
        self.alphas = alphas

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

        if self.alphas is None:
            if self.n_alphas < 1:
                raise ValueError("n_alphas should be strictly positive.")
            # Set values of alpha for regularization
            alphas = _alpha_grid(X, y, l1_ratio=1, n_alphas=self.n_alphas,
                                 eps=self.eps)
        else:
            alphas = self.alphas

        # init cross-validation generator
        cv = check_cv(self.cv, classifier=False)

        cv_paths = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_relasso_path_residues)(
                X[train], y[train], X[test], y[test], copy=False,
                verbose=max(0, self.verbose - 1), max_iter=self.max_iter,
                alphas=alphas)
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
                  theta=self.theta_)

        return self
