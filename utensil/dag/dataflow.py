from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List


from utensil.dag.helper import warn_left_keys

try:
    import xgboost
    import numpy as np
    import pandas as pd
    import sklearn.datasets
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
        model = self._model.train(dataset.features, dataset.target)
        return SklearnModel(model)

    def predict(self, features: Features) -> Target:
        return Target(self._model.predict(features))


@dataclass
class NodeProcess:
    params: Dict[str, Any] = field(repr=False)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class LoadData(NodeProcess):
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
            features, target = sklearn.datasets.load_svmlight_file(self.url)
        else:
            raise ValueError(self.dformat)
        features = pd.DataFrame.sparse.from_spmatrix(features).loc[:, self.features.keys()].rename(
            columns=self.features)
        target = pd.Series(target, name=self.target)
        return Dataset(Target(target), Features(features))


@dataclass
class FilterRows(NodeProcess):
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
class MakeDataset(NodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, target: Target, features: Features) -> Dataset:
        print(Dataset(target, features))
        return Dataset(target, features)


@dataclass
class GetTarget(NodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, dataset: Dataset) -> Target:
        return dataset.target


@dataclass
class GetFeature(NodeProcess):
    feature: str = None

    def __post_init__(self):
        self.feature = self.params.pop('FEATURE')
        warn_left_keys(self.params)
        del self.params

    def __call__(self, dataset: Dataset) -> Feature:
        return Feature(dataset.features.loc[:, self.feature])


@dataclass
class MergeFeatures(NodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, *features: Feature) -> Features:
        return Features(pd.concat([pd.Series(f) for f in features], axis=1))


def is_number(n):
    # check whether n is a number
    try:
        float(n)
        return True
    except Exception:
        return False


@dataclass
class LinearNormalize(NodeProcess):
    upper: Dict[str, Any] = None
    lower: Dict[str, Any] = None

    def __post_init__(self):
        self.upper = self.params.pop('UPPER', None)
        self.lower = self.params.pop('LOWER', None)
        warn_left_keys(self.params)
        del self.params

    def __call__(self, arr1d: np.ndarray) -> np.ndarray:
        if self.upper is not None:
            hifrom = self.upper.pop('FROM')
            hito = self.upper.pop('TO')
            if hifrom == 'MAX':
                hifrom = np.max(arr1d)
            elif hifrom == 'MIN':
                hifrom = np.min(arr1d)
            else:
                raise ValueError(hifrom)
            if is_number(hito):
                hito = float(hito)
            else:
                raise ValueError(hito)
        else:
            hifrom = np.max(arr1d)
            hito = np.max(arr1d)

        if self.lower is not None:
            lofrom = self.lower.pop('FROM')
            loto = self.lower.pop('TO')
            if lofrom == 'MAX':
                lofrom = np.max(arr1d)
            elif lofrom == 'MIN':
                lofrom = np.min(arr1d)
            else:
                raise ValueError(lofrom)
            if is_number(loto):
                loto = float(loto)
            else:
                raise ValueError(loto)
        else:
            lofrom = np.min(arr1d)
            loto = np.min(arr1d)

        return arr1d * (hito - loto) / (hifrom - lofrom) + (hito * lofrom - hifrom - loto) / (hifrom - lofrom)


@dataclass
class MakeModel(NodeProcess):
    method: str = None

    def __post_init__(self):
        self.method = self.params.pop('METHOD')
        warn_left_keys(self.params)
        del self.params

    def __call__(self):
        if self.method == 'XGBOOST':
            return SklearnModel(xgboost.XGBRegressor())
        else:
            raise ValueError


@dataclass
class Train(NodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, model: Model, dataset: Dataset) -> Model:
        model = model.train(dataset)
        return model


@dataclass
class Predict(NodeProcess):
    def __post_init__(self):
        warn_left_keys(self.params)
        del self.params

    def __call__(self, model: Model, features: Features) -> Target:
        model.predict(features)