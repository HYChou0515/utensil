import os
import sys
import string
from test.fixtures import FIXTURE_BASE

import pytest

try:
    import numpy as np
    import pandas as pd
    import xgboost
    from sklearn import datasets, ensemble

    from utensil.loopflow.functions import dataflow
except ImportError:
    pytestmark = pytest.mark.skip

if sys.platform == "darwin":
    pytestmark = pytest.mark.skip


@pytest.fixture(name='heart_scale')
def fixture_heart_scale():
    path = os.path.join(FIXTURE_BASE, 'heart_scale')
    features, target, *_ = datasets.load_svmlight_file(path)
    features = (pd.DataFrame.sparse.from_spmatrix(features).rename(columns={
        i: string.ascii_lowercase[i] for i in range(features.shape[1])
    }))
    target = pd.Series(target, name='label')
    return dataflow.Dataset(dataflow.Target(target),
                            dataflow.Features(features))


def test_TestLoadData_file_url(heart_scale):
    path = os.path.join(FIXTURE_BASE, 'heart_scale')
    load_data = dataflow.LoadData(dformat='SVMLIGHT',
                                  url=f'file://{path}',
                                  features={
                                      0: 'a',
                                      1: 'b'
                                  },
                                  target='target')
    data = load_data.main()
    assert data.nrows == heart_scale.nrows
    assert data.ncols == 2


def test_TestLoadData_unknown_url_scheme():
    load_data = dataflow.LoadData(dformat='SVMLIGHT',
                                  url="other-scheme://path/to/data",
                                  features={
                                      0: 'a',
                                      1: 'b'
                                  },
                                  target='target')
    with pytest.raises(ValueError,
                       match='scheme "other-scheme" cannot be recognized in '
                       'other-scheme://path/to/data'):
        load_data.main()


def test_TestLoadData_unknown_dformat():
    load_data = dataflow.LoadData(dformat='OTHER_FORMAT',
                                  url="other-scheme://path/to/data",
                                  features={
                                      0: 'a',
                                      1: 'b'
                                  },
                                  target='target')
    with pytest.raises(ValueError,
                       match='data format "OTHER_FORMAT" cannot be recognized'):
        load_data.main()


def test_FilterRows_unknown_filter(heart_scale):
    filter_rows = dataflow.FilterRows(filter_by={'a': [0, 1]})
    with pytest.raises(ValueError, match='filter key "a" cannot be recognized'):
        filter_rows.main(heart_scale)


def test_SamplingRows_ratio_default():
    sampling_rows = dataflow.SamplingRows()
    assert sampling_rows.ratio == 1.0


def test_SamplingRows_raiseValueError_when_NotReplaceAndSamplingNumbersTooLarge(
        heart_scale):
    sampling_rows = dataflow.SamplingRows(number=heart_scale.nrows + 1,
                                          replace=False)
    with pytest.raises(
            ValueError,
            match=("sampling number should at most the same size as the "
                   "dataset "
                   'when "replace" is set False')):
        sampling_rows.main(heart_scale)


@pytest.mark.parametrize('number', [0, 1, 1000])
@pytest.mark.parametrize('ratio', [0.8, 1.0, 1.2])
@pytest.mark.parametrize('use_number', [True, False])
def test_SamplingRows_with_NotStratified(heart_scale, number, ratio,
                                         use_number):
    if use_number:
        sampling_rows = dataflow.SamplingRows(number=number,
                                              stratified=False,
                                              replace=True)
        expected_nrows = number
    else:
        sampling_rows = dataflow.SamplingRows(ratio=ratio,
                                              stratified=False,
                                              replace=True)
        expected_nrows = int(ratio * heart_scale.nrows)
    data = sampling_rows.main(heart_scale)
    assert data.nrows == expected_nrows
    assert data.ncols == heart_scale.ncols


@pytest.mark.parametrize('min_plus', [-1, 0, 1])
def test_SamplingRows_with_Stratified(heart_scale, min_plus):
    vc = heart_scale.target.value_counts()
    nr_cat = vc.shape[0]
    min_cnt = vc.min()
    sampling_rows = dataflow.SamplingRows(number=(min_cnt + min_plus) * nr_cat,
                                          stratified=True,
                                          replace=False)
    data = sampling_rows.main(heart_scale)
    assert data.nrows == (min_cnt + min_plus) * nr_cat
    assert data.ncols == heart_scale.ncols
    assert data.target.value_counts().min() == min(min_cnt, min_cnt + min_plus)


def test_MakeDataset(heart_scale):
    make_dataset = dataflow.MakeDataset()
    dataset = make_dataset.main(heart_scale.target, heart_scale.features)
    pd.testing.assert_series_equal(dataset.target, heart_scale.target)
    pd.testing.assert_frame_equal(dataset.features, heart_scale.features)


def test_get_max():
    arr = np.arange(10)
    m = dataflow._get_max(arr)  # pylint: disable=protected-access
    assert all(arr <= m)


def test_get_min():
    arr = np.arange(10)
    m = dataflow._get_min(arr)  # pylint: disable=protected-access
    assert all(arr >= m)


@pytest.mark.parametrize('upper_from', [-10, 1, 10])
@pytest.mark.parametrize('upper_to', [-10, 1, 10])
@pytest.mark.parametrize('lower_from', [-10, 1, 10])
@pytest.mark.parametrize('lower_to', [-10, 1, 10])
def test_LinearNormalize_ArithmeticSequence(upper_from, upper_to, lower_from,
                                            lower_to):
    upper = {'FROM': upper_from, 'TO': upper_to}
    lower = {'FROM': lower_from, 'TO': lower_to}
    linear_normalize = dataflow.LinearNormalize(upper=upper, lower=lower)
    arr = np.arange(10)
    if upper_from == lower_from and upper_to != lower_to:
        with pytest.raises(
                ValueError,
                match=
                'Cannot map a single value (.*) to a different values (.*, .*)'
        ):
            linear_normalize.main(arr)
    else:
        norm_arr = linear_normalize.main(arr)
        if upper_from == lower_from and upper_to == lower_to:
            assert all(arr == norm_arr)

        d = np.diff(norm_arr)
        assert np.allclose(
            d,
            d[0]), 'arr is an arithmetic sequence, so should normalized arr be'


def test_LinearNormalize():
    upper = {'FROM': 4, 'TO': 10}
    lower = {'FROM': 0, 'TO': -10}
    linear_normalize = dataflow.LinearNormalize(upper=upper, lower=lower)
    arr = np.arange(10)
    norm_arr = linear_normalize.main(arr)
    assert np.allclose(np.arange(-10, 40, 5), norm_arr)


@pytest.mark.parametrize('upper_from', ['MIN', 'MAX', -10, 1, 10])
@pytest.mark.parametrize('upper_to', ['MIN', 'MAX', -10, 1, 10])
@pytest.mark.parametrize('lower_from', ['MIN', 'MAX', -10, 1, 10])
@pytest.mark.parametrize('lower_to', ['MIN', 'MAX', -10, 1, 10])
def test_LinearNormalize_MinMax(upper_from, upper_to, lower_from, lower_to):
    upper = {'FROM': upper_from, 'TO': upper_to}
    lower = {'FROM': lower_from, 'TO': lower_to}
    if 'MAX' not in upper.values() and 'MIN' not in upper.values() and (
            'MAX' not in lower.values() and 'MIN' not in lower.values()):
        return
    linear_normalize = dataflow.LinearNormalize(upper=upper, lower=lower)
    arr = np.arange(10)
    if upper_from == lower_from and upper_to != lower_to:
        with pytest.raises(
                ValueError,
                match=
                'Cannot map a single value (.*) to a different values (.*, .*)'
        ):
            linear_normalize.main(arr)
        return
    norm_arr = linear_normalize.main(arr)

    arr_max = max(arr)
    arr_min = min(arr)
    upper = {
        k: arr_max if v == 'MAX' else arr_min if v == 'MIN' else v
        for k, v in upper.items()
    }
    lower = {
        k: arr_max if v == 'MAX' else arr_min if v == 'MIN' else v
        for k, v in lower.items()
    }
    linear_normalize = dataflow.LinearNormalize(upper=upper, lower=lower)
    expected = linear_normalize.main(arr)
    assert np.allclose(norm_arr, expected)


def test_LinearNormalize_raiseValueError_when_commandNotFound():
    linear_normalize = dataflow.LinearNormalize(upper={'FROM': 'SOME_CMD'})
    with pytest.raises(ValueError,
                       match='command "SOME_CMD" cannot be recognized'):
        linear_normalize.main(np.arange(10))


def test_LinearNormalize_raiseValueError_when_badLimit():
    linear_normalize = dataflow.LinearNormalize(upper={'FROM': [1]})
    with pytest.raises(ValueError, match='Expecting str or number, got list'):
        linear_normalize.main(np.arange(10))


@pytest.mark.parametrize('method', [
    'XGBOOST_REGRESSOR', 'XGBOOST_CLASSIFIER',
    'SKLEARN_GRADIENT_BOOSTING_CLASSIFIER'
])
@pytest.mark.parametrize('learning_rate', [0.1, None])
@pytest.mark.parametrize('max_depth', [3, None])
@pytest.mark.parametrize('n_estimators', [5, None])
def test_MakeModel(method, learning_rate, max_depth, n_estimators):
    make_model = dataflow.MakeModel(method)
    model_params = {
        'LEARNING_RATE':
            dataflow.MISSING if learning_rate is None else learning_rate,
        'MAX_DEPTH':
            dataflow.MISSING if max_depth is None else max_depth,
        'N_ESTIMATORS':
            dataflow.MISSING if n_estimators is None else n_estimators,
    }
    model = make_model.main(model_params)
    assert isinstance(model, dataflow.Model)
    if method == 'XGBOOST_REGRESSOR':
        assert isinstance(model._model, xgboost.XGBRegressor)  # pylint: disable=protected-access
    if method == 'XGBOOST_CLASSIFIER':
        assert isinstance(model._model, xgboost.XGBClassifier)  # pylint: disable=protected-access
    if method == 'SKLEARN_GRADIENT_BOOSTING_CLASSIFIER':
        assert isinstance(model._model, ensemble.GradientBoostingClassifier)  # pylint: disable=protected-access


def test_MakeModel_raiseValueError_when_nonSupportedMethod():
    make_model = dataflow.MakeModel('OTHER_METHOD')
    with pytest.raises(ValueError,
                       match='method "OTHER_METHOD" cannot be recognized'):
        make_model.main({})


def test_Predict(heart_scale):
    import warnings
    model = xgboost.XGBClassifier(n_estimators=3, max_depth=3)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", category=UserWarning)
        model = model.fit(heart_scale.features, heart_scale.target)
    model = dataflow.SklearnModel(model)
    prediction = model.predict(heart_scale.features)
    assert sum(prediction == heart_scale.target) / prediction.shape[0] > 0.7
