from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

from utensil.general import warn_left_keys

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    raise e


class Feature(pd.Series):
    pass


class Features(pd.DataFrame):
    pass


class Target(pd.Series):
    pass


@dataclass
class Dataset:
    target: Target
    features: Features


class Model:
    def train(self, dataset: Dataset) -> Model:
        return NotImplemented

    def predict(self, features: Features) -> Target:
        return NotImplemented


class SklearnModel(Model):
    def __init__(self, model):
        self._model = model

    def train(self, dataset: Dataset) -> Model:
        model = self._model.fit(dataset.features, dataset.target)
        return SklearnModel(model)

    def predict(self, features: Features) -> Target:
        return Target(self._model.predict(features))


@dataclass
class NodeProcess:
    params: Dict[str, Any] = field(repr=False)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class StatefulNodeProcess(NodeProcess):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class StatelessNodeProcess(NodeProcess):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class LoadData(StatelessNodeProcess):
    dformat: str = None
    url: str = None
    target: str = None
    features: Dict[int, str] = None

    def __post_init__(self):
        self.dformat = self.params.pop('FORMAT', None)
        self.url = self.params.pop('URL', None)
        self.target = self.params.pop('TARGET', None)
        self.features = self.params.pop('FEATURES', None)
        warn_left_keys(self.params)
        del self.params

    def __call__(self) -> Dataset:
        if self.dformat == 'SVMLIGHT':
            try:
                import sklearn.datasets
            except ImportError as e:
                raise e
            features, target = sklearn.datasets.load_svmlight_file(self.url)
        else:
            raise ValueError(self.dformat)
        features = pd.DataFrame.sparse.from_spmatrix(features).loc[:, self.features.keys()].rename(
            columns=self.features)
        target = pd.Series(target, name=self.target)
        return Dataset(Target(target), Features(features))


@dataclass
class FilterRows(StatelessNodeProcess):
    filter_by: Dict[str, Any] = None

    def __post_init__(self):
        self.filter_by = self.params.pop('FILTER_BY', None)
        warn_left_keys(self.params)
        del self.params

    def __call__(self, dataset: Dataset) -> Dataset:
        idx = dataset.target.index
        for filter_key, filter_val in self.filter_by.items():
            if filter_key == 'TARGET':
                idx = idx.intersection(dataset.target.index[dataset.target.isin(filter_val)])
            else:
                raise ValueError(filter_key)
        dataset.target = dataset.target.loc[idx]
        dataset.features = dataset.features.loc[idx]
        return dataset


@dataclass
class MakeDataset(StatelessNodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, target: Target, features: Features) -> Dataset:
        return Dataset(target, features)


@dataclass
class GetTarget(StatelessNodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, dataset: Dataset) -> Target:
        return dataset.target


@dataclass
class GetFeature(StatelessNodeProcess):
    feature: str = None

    def __post_init__(self):
        self.feature = self.params.pop('FEATURE')
        warn_left_keys(self.params)
        del self.params

    def __call__(self, dataset: Dataset) -> Feature:
        return Feature(dataset.features.loc[:, self.feature])


@dataclass
class MergeFeatures(StatelessNodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, *features: Feature) -> Features:
        return Features(pd.concat([pd.Series(f) for f in features], axis=1))


def is_number(n):
    return pd.api.types.is_number(n)


def get_max(arr):
    if pd.api.types.is_sparse(arr):
        if arr.sparse.sp_values.shape[0] < arr.shape[0]:
            return max(arr.sparse.fill_value, np.max(arr.sparse.sp_values))
        else:
            return np.max(arr.sparse.sp_values)
    else:
        return np.max(arr)


def get_min(arr):
    if pd.api.types.is_sparse(arr):
        if arr.sparse.sp_values.shape[0] < arr.shape[0]:
            return min(arr.sparse.fill_value, np.min(arr.sparse.sp_values))
        else:
            return np.min(arr.sparse.sp_values)
    else:
        return np.min(arr)


@dataclass
class LinearNormalize(StatelessNodeProcess):
    upper: Dict[str, Any] = None
    lower: Dict[str, Any] = None

    @staticmethod
    def _compile_limit(cmd, arr1d):
        if isinstance(cmd, str):
            if cmd == 'MAX':
                return get_max(arr1d)
            elif cmd == 'MIN':
                return get_min(arr1d)
        elif is_number(cmd):
            return cmd
        raise ValueError(cmd)

    def __post_init__(self):
        self.upper = {'FROM': 'MAX', 'TO': 'MAX'}
        self.upper.update(**self.params.pop('UPPER', {}))
        self.lower = {'FROM': 'MIN', 'TO': 'MIN'}
        self.lower.update(**self.params.pop('LOWER', {}))
        warn_left_keys(self.params)
        del self.params

    def __call__(self, arr1d: np.ndarray) -> np.ndarray:
        hifrom = self._compile_limit(self.upper.pop('FROM'), arr1d)
        hito = self._compile_limit(self.upper.pop('TO'), arr1d)
        lofrom = self._compile_limit(self.lower.pop('FROM'), arr1d)
        loto = self._compile_limit(self.lower.pop('TO'), arr1d)

        return arr1d * (hito - loto) / (hifrom - lofrom) + (hito * lofrom - hifrom - loto) / (hifrom - lofrom)


@dataclass
class MakeModel(StatelessNodeProcess):
    method: str = None

    def __post_init__(self):
        self.method = self.params.pop('METHOD')
        warn_left_keys(self.params)
        del self.params

    def __call__(self):
        if self.method == 'XGBOOST_REGRESSOR':
            try:
                import xgboost
            except ImportError as e:
                raise e
            return SklearnModel(xgboost.XGBRegressor())
        else:
            raise ValueError


@dataclass
class Train(StatelessNodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, model: Model, dataset: Dataset) -> Model:
        model = model.train(dataset)
        return model


@dataclass
class Predict(StatelessNodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, model: Model, features: Features) -> Target:
        return model.predict(features)
