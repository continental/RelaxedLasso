import pytest

from sklearn.utils.estimator_checks import check_estimator

from relaxed_lasso import RelaxedLassoLars
from relaxed_lasso import RelaxedLassoLarsCV


@pytest.mark.parametrize(
    "Estimator", [RelaxedLassoLars(), RelaxedLassoLarsCV()]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
