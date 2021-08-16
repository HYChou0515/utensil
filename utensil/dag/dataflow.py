from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Generator, Tuple, List, Type, Union

from utensil.general import warn_left_keys
from utensil.random_search import RandomizedParam

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    raise e


class MISSING:
    pass


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
    state: Any = None

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

    def __call__(self, model_params: Dict[str, Any]):
        if self.method == 'XGBOOST_REGRESSOR':
            try:
                import xgboost
            except ImportError as e:
                raise e
            return SklearnModel(xgboost.XGBRegressor())
        elif self.method == 'XGBOOST_CLASSIFIER':
            try:
                import xgboost
            except ImportError as e:
                raise e
            _model_params = {
                'learning_rate': model_params.pop('LEARNING_RATE', MISSING),
                'max_depth': model_params.pop('MAX_DEPTH', MISSING),
                'n_estimators': model_params.pop('N_ESTIMATORS', MISSING),
            }
            for k, v in list(_model_params.items()):
                if v is MISSING:
                    del _model_params[k]
            warn_left_keys(model_params)
            return SklearnModel(xgboost.XGBClassifier(**_model_params, use_label_encoder=True))
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


@dataclass
class ParameterSearch(StatefulNodeProcess):
    search_map: Dict[str, Any] = None
    nr_randomized_params: int = field(init=False)
    rng: np.random.Generator = field(init=False)

    seed_gen: Generator[Tuple[int, Tuple]] = field(init=False)

    def __post_init__(self):
        self.state = self.params.pop('INIT_STATE', 0)
        self.nr_randomized_params = 0
        self.rng = np.random.default_rng(self.params.pop('SEED', 0))
        self.seed_gen = self._generate_seed()
        self.search_map = {}
        for param_name, search_method in self.params.pop('SEARCH_MAP', {}).items():
            if isinstance(search_method, dict):
                if len(search_method) != 1:
                    raise ValueError
                search_type, search_option = search_method.popitem()
                self.search_map[param_name] = RandomizedParam.create_randomized_param(search_type, search_option)
                self.nr_randomized_params += 1
            else:
                self.search_map[param_name] = search_method

        warn_left_keys(self.params)
        del self.params

    def __call__(self):
        state, r_list = next(self.seed_gen)
        if len(r_list) != self.nr_randomized_params:
            raise ValueError
        r = iter(r_list)
        params = {}
        for k, v in self.search_map.items():
            if isinstance(v, RandomizedParam):
                params[k] = v.from_random(next(r))
            else:
                params[k] = v
        return params

    def _random_between(self, a, b, **kwargs):
        return self.rng.random(**kwargs) * (b - a) + a

    def _generate_seed(self):
        rand_space = []
        while True:
            base = 2 ** int(np.log2(self.state+1))
            offset = self.state+1-base
            if offset == 0 or len(rand_space) == 0:
                linspace = np.linspace(0, 1, base+1)
                rand_space = np.array([self._random_between(
                    linspace[i], linspace[i+1], size=self.nr_randomized_params
                ) for i in range(base)])

                for i in range(self.nr_randomized_params):
                    self.rng.shuffle(rand_space[:, i])

            model_r = tuple(rand_space[offset])

            yield self.state, model_r
            self.state += 1


@dataclass
class Score(StatelessNodeProcess):
    methods: List[str] = None

    def __post_init__(self):
        if isinstance(self.params, str):
            self.methods = [self.params]
        elif isinstance(self.params, list):
            self.methods = self.params
        else:
            raise TypeError
        del self.params

    def __call__(self, prediction: Target, ground_truth: Target):
        ret = []
        for method in self.methods:
            if method == 'ACCURACY':
                ret.append((method, np.sum(prediction.values == ground_truth.values) / prediction.shape[0]))
        return ret


@dataclass
class ChangeTypeTo(StatelessNodeProcess):
    to_type: Type = None

    def __post_init__(self):
        if isinstance(self.params, str):
            if self.params == 'INTEGER':
                self.to_type = int
            elif self.params == 'FLOAT':
                self.to_type = float
            else:
                raise ValueError
        else:
            raise TypeError
        del self.params

    def __call__(self, arr: Union[Feature, Target]):
        return arr.astype(self.to_type)
