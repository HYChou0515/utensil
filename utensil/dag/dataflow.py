from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from utensil.dag.dag import NodeProcessFunction
from utensil.general import warn_left_keys
from utensil.random_search import RandomizedParam

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    raise e


class _MISSING:
    pass


MISSING = _MISSING()


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

    @property
    def nrows(self):
        if self.target.shape[0] != self.features.shape[0]:
            raise ValueError("rows of target and that of features should be the same")
        return self.target.shape[0]

    @property
    def ncols(self):
        return self.features.shape[1]


class Model:
    def train(self, dataset: Dataset) -> Model:
        raise NotImplementedError

    def predict(self, features: Features) -> Target:
        raise NotImplementedError


class SklearnModel(Model):
    def __init__(self, model):
        self._model = model

    def train(self, dataset: Dataset) -> Model:
        model = self._model.fit(dataset.features, dataset.target)
        return SklearnModel(model)

    def predict(self, features: Features) -> Target:
        return Target(self._model.predict(features))


class LoadData(NodeProcessFunction):
    def __init__(self, dformat: str, url: str, target: str, features: Dict[int, str]):
        super(self.__class__, self).__init__()
        self.dformat = dformat
        self.url = url
        self.target = target
        self.features = features

    def main(self) -> Dataset:
        if self.dformat == "SVMLIGHT":
            try:
                import sklearn.datasets
            except ImportError as e:
                raise e
            features, target = sklearn.datasets.load_svmlight_file(self.url)
        else:
            raise ValueError(self.dformat)
        features = (
            pd.DataFrame.sparse.from_spmatrix(features)
            .loc[:, self.features.keys()]
            .rename(columns=self.features)
        )
        target = pd.Series(target, name=self.target)
        return Dataset(Target(target), Features(features))


class FilterRows(NodeProcessFunction):
    def __init__(self, filter_by: Dict[str, Any]):
        super(self.__class__, self).__init__()
        self.filter_by = filter_by

    def main(self, dataset: Dataset) -> Dataset:
        idx = dataset.target.index
        for filter_key, filter_val in self.filter_by.items():
            if filter_key == "TARGET":
                idx = idx.intersection(
                    dataset.target.index[dataset.target.isin(filter_val)]
                )
            else:
                raise ValueError(filter_key)
        dataset.target = dataset.target.loc[idx]
        dataset.features = dataset.features.loc[idx]
        return dataset


class SamplingRows(NodeProcessFunction):
    def __init__(
        self,
        number: int = MISSING,
        ratio: float = MISSING,
        stratified: bool = False,
        replace: bool = False,
        random_seed: Union[bool, int] = 0,
        return_rest: bool = False,
    ):
        super(self.__class__, self).__init__()
        self.number = number
        self.ratio = ratio
        self.stratified = stratified
        self.replace = replace
        self.random_seed = random_seed
        self.return_rest = return_rest

        if self.ratio is MISSING and self.number is MISSING:
            self.ratio = 1.0

    def _get_number_each(self, value_counts: pd.Series, ttl_number: int):
        if self.replace or ttl_number // value_counts.shape[0] <= value_counts.min():
            number_each = (
                pd.Series(0, index=value_counts.index)
                + ttl_number // value_counts.shape[0]
            )
            categories = number_each.index.to_numpy(copy=True)
            self._rng.shuffle(categories)
            residual = ttl_number - number_each.sum()
            number_each += pd.Series(
                {cat: 1 if i < residual else 0 for i, cat in enumerate(categories)}
            )
        else:
            number_each = pd.Series(value_counts.min(), index=value_counts.index)
            residual = self._get_number_each(
                value_counts.drop(labels=value_counts.idxmin()) - value_counts.min(),
                ttl_number - np.sum(number_each),
            )
            number_each += (residual + pd.Series(0, index=number_each.index)).astype(
                int
            )
        return number_each

    def main(self, dataset: Dataset) -> Union[Dataset, Dict[str, Dataset]]:
        if self.stratified:
            ttl_number = (
                int(self.ratio * dataset.nrows)
                if self.ratio is not MISSING
                else self.number
            )
            value_counts = dataset.target.value_counts()
            self._rng = np.random.default_rng(self.random_seed)
            if not self.replace and ttl_number > dataset.nrows:
                raise ValueError(
                    "sampling number should at most the same size as the dataset "
                    'when "replace" is set False'
                )
            number_each = self._get_number_each(value_counts, ttl_number)
            selected_idx = []
            for cat, idx in dataset.target.groupby(dataset.target).groups.items():
                selected_idx.append(
                    self._rng.choice(idx, number_each[cat], replace=self.replace)
                )
            selected_idx = np.concatenate(selected_idx)
            imap = {idx: i for i, idx in enumerate(dataset.target.index)}
            selected_idx = np.array(sorted(selected_idx, key=imap.__getitem__))
            new_target = dataset.target.loc[selected_idx]
        elif self.ratio is not MISSING:
            new_target = dataset.target.sample(
                frac=self.ratio, replace=self.replace, random_state=self.random_seed
            )
        else:
            new_target = dataset.target.sample(
                n=self.number, replace=self.replace, random_state=self.random_seed
            )
        new_features = dataset.features.loc[new_target.index]
        if self.return_rest:
            rest_index = dataset.target.index.difference(new_target.index)
            rest_target = dataset.target.loc[rest_index]
            rest_features = dataset.features.loc[rest_index]
            return {
                "sampled": Dataset(Target(new_target), Features(new_features)),
                "rest": Dataset(Target(rest_target), Features(rest_features)),
            }
        else:
            return Dataset(Target(new_target), Features(new_features))


class MakeDataset(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, target: Target, features: Features) -> Dataset:
        return Dataset(target, features)


class GetTarget(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, dataset: Dataset) -> Target:
        return dataset.target


class GetFeature(NodeProcessFunction):
    def __init__(self, feature: str):
        super(self.__class__, self).__init__()
        self.feature = feature

    def main(self, dataset: Dataset) -> Feature:
        return Feature(dataset.features.loc[:, self.feature])


class MergeFeatures(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, *features: Feature) -> Features:
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


class LinearNormalize(NodeProcessFunction):
    def __init__(self, upper: Dict[str, Any] = None, lower: Dict[str, Any] = None):
        super(self.__class__, self).__init__()
        self.upper = {"FROM": "MAX", "TO": "MAX"}
        self.upper.update({} if upper is None else upper)
        self.lower = {"FROM": "MIN", "TO": "MIN"}
        self.lower.update({} if lower is None else lower)

    @staticmethod
    def _compile_limit(cmd, arr1d):
        if isinstance(cmd, str):
            if cmd == "MAX":
                return get_max(arr1d)
            elif cmd == "MIN":
                return get_min(arr1d)
        elif is_number(cmd):
            return cmd
        raise ValueError(cmd)

    def main(self, arr1d: np.ndarray) -> np.ndarray:
        hifrom = self._compile_limit(self.upper.pop("FROM"), arr1d)
        hito = self._compile_limit(self.upper.pop("TO"), arr1d)
        lofrom = self._compile_limit(self.lower.pop("FROM"), arr1d)
        loto = self._compile_limit(self.lower.pop("TO"), arr1d)

        return arr1d * (hito - loto) / (hifrom - lofrom) + (
            hito * lofrom - hifrom - loto
        ) / (hifrom - lofrom)


class MakeModel(NodeProcessFunction):
    def __init__(self, method):
        super(self.__class__, self).__init__()
        self.method = method

    @staticmethod
    def after_assign_params_routine(_from, _to):
        for k, v in list(_to.items()):
            if v is MISSING:
                del _to[k]
        warn_left_keys(_from)

    def main(self, model_params: Dict[str, Any]):
        if self.method == "XGBOOST_REGRESSOR":
            try:
                import xgboost
            except ImportError as e:
                raise e
            return SklearnModel(xgboost.XGBRegressor())
        elif self.method == "XGBOOST_CLASSIFIER":
            try:
                import xgboost
            except ImportError as e:
                raise e
            _model_params = {
                "learning_rate": model_params.pop("LEARNING_RATE", MISSING),
                "max_depth": model_params.pop("MAX_DEPTH", MISSING),
                "n_estimators": model_params.pop("N_ESTIMATORS", MISSING),
            }
            self.after_assign_params_routine(model_params, _model_params)
            return SklearnModel(
                xgboost.XGBClassifier(**_model_params, use_label_encoder=True)
            )
        elif self.method == "SKLEARN_GRADIENT_BOOSTING_CLASSIFIER":
            try:
                from sklearn.ensemble import GradientBoostingClassifier
            except ImportError as e:
                raise e
            _model_params = {
                "learning_rate": model_params.pop("LEARNING_RATE", MISSING),
                "max_depth": model_params.pop("MAX_DEPTH", MISSING),
                "n_estimators": model_params.pop("N_ESTIMATORS", MISSING),
            }
            self.after_assign_params_routine(model_params, _model_params)
            return SklearnModel(GradientBoostingClassifier(**_model_params))
        else:
            raise ValueError


class Train(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, model: Model, dataset: Dataset) -> Model:
        model = model.train(dataset)
        return model


class Predict(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, model: Model, features: Features) -> Target:
        return model.predict(features)


class ParameterSearch(NodeProcessFunction):
    def __init__(self, init_state=0, seed: int = 0, search_map: Dict = None):
        super(self.__class__, self).__init__()
        self.state = init_state
        self._nr_randomized_params = 0
        self._rng = np.random.default_rng(seed)
        self._seed_gen = self._generate_seed()
        self._search_map = {}
        search_map = {} if search_map is None else search_map
        for param_name, search_method in search_map.items():
            if isinstance(search_method, dict):
                if len(search_method) != 1:
                    raise ValueError
                search_type, search_option = search_method.popitem()
                self._search_map[param_name] = RandomizedParam.create_randomized_param(
                    search_type, search_option
                )
                self._nr_randomized_params += 1
            else:
                self._search_map[param_name] = search_method

    def _random_between(self, a, b, **kwargs):
        return self._rng.random(**kwargs) * (b - a) + a

    def _generate_seed(self):
        rand_space = []
        while True:
            base = 2 ** int(np.log2(self.state + 1))
            offset = self.state + 1 - base
            if offset == 0 or len(rand_space) == 0:
                linspace = np.linspace(0, 1, base + 1)
                rand_space = np.array(
                    [
                        self._random_between(
                            linspace[i],
                            linspace[i + 1],
                            size=self._nr_randomized_params,
                        )
                        for i in range(base)
                    ]
                )

                for i in range(self._nr_randomized_params):
                    self._rng.shuffle(rand_space[:, i])

            model_r = tuple(rand_space[offset])

            yield self.state, model_r
            self.state += 1

    def main(self):
        state, r_list = next(self._seed_gen)
        if len(r_list) != self._nr_randomized_params:
            raise ValueError
        r = iter(r_list)
        params = {}
        for k, v in self._search_map.items():
            if isinstance(v, RandomizedParam):
                params[k] = v.from_random(next(r))
            else:
                params[k] = v
        return params


class Score(NodeProcessFunction):
    def __init__(self, dataset: str = MISSING, methods: Union[str, List[str]] = None):
        super(self.__class__, self).__init__()
        self.dataset_name = dataset
        self.methods = [] if methods is None else methods
        if isinstance(self.methods, str):
            self.methods = [self.methods]

    def main(
        self,
        prediction: Union[Target, Features, Dataset],
        ground_truth: Union[Target, Dataset],
        model: Model,
    ):
        if isinstance(prediction, Target):
            pass
        elif isinstance(prediction, Features):
            prediction = model.predict(prediction)
        elif isinstance(prediction, Dataset):
            prediction = model.predict(prediction.features)
        else:
            raise TypeError

        if isinstance(ground_truth, Target):
            pass
        elif isinstance(ground_truth, Dataset):
            ground_truth = ground_truth.target
        else:
            raise TypeError

        def wrap_output(_method, _score):
            if self.dataset_name is MISSING:
                return _method, _score
            else:
                return _method, self.dataset_name, _score

        ret = []
        for method in self.methods:
            if method == "ACCURACY":
                ret.append(
                    wrap_output(
                        method,
                        np.sum(prediction.values == ground_truth.values)
                        / prediction.shape[0],
                    )
                )
        return ret


class ChangeTypeTo(NodeProcessFunction):
    def __init__(self, to_type: str):
        super(self.__class__, self).__init__()
        if to_type == "INTEGER":
            self.to_type = int
        elif to_type == "FLOAT":
            self.to_type = float
        else:
            raise RuntimeError("E22")

    def main(self, arr: Union[Feature, Target]):
        return arr.astype(self.to_type)


from utensil.dag.dag import Dag

dag_path = "../../test/dag/covtype.dag"
dag = Dag.parse_yaml(dag_path)
dag.start()
