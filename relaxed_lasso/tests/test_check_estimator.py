import pytest

from sklearn.utils.estimator_checks import check_estimator

from relaxed_lasso import RelaxedLassoLars
from relaxed_lasso import RelaxedLassoLarsCV
from relaxed_lasso import RelaxedLasso
from relaxed_lasso import RelaxedLassoCV


@pytest.mark.parametrize(
    "Estimator", [RelaxedLassoLars(), RelaxedLassoLarsCV(),
                  RelaxedLasso(), RelaxedLassoCV()]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
