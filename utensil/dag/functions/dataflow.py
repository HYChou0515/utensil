"""**Provide NodeProcessFunction for machine learning work flows.**

Example:

.. highlight:: python
.. code-block:: python

    from utensil.dag.functions import dataflow
    from utensil.dag.dag import register_node_process_functions
    register_node_process_functions(dataflow)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from utensil import get_logger
from utensil.dag.dag import NodeProcessFunction
from utensil.dag.functions.basic import MISSING
from utensil.general import warn_left_keys
from utensil.random_search import RandomizedParam

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    raise e

try:
    import sklearn.datasets as sklearn_datasets
except ImportError as e:
    sklearn_datasets = e

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError as e:
    GradientBoostingClassifier = e

try:
    import xgboost as _xgboost
except ImportError as e:
    _xgboost = e

logger = get_logger(__name__)


class Feature(pd.Series):
    """A feature of a dataset.

    :class:`.Feature` is an individual measurable property or characteristic
    of a phenomenon. It can be a list of numbers, strings with or without
    missing values. The length of a feature (missing values included) should
    be the number of instance in a dataset.
    """


class Features(pd.DataFrame):
    """A list of features.

    :class:`.Features` is a list of :class:`.Feature`. It can be represented as
    a matrix of numbers, strings, missing values, etc.
    """


class Target(pd.Series):
    """The target of a dataset.

    :class:`.Target` is whatever the output of the input variables. Typically,
    it is the variables a supervised model trying to learn to predict,
    either numerical or categorical.
    """


@dataclass
class Dataset:
    """A dataset used to train a model or to let a model predict its target.

    A pair of :class:`.Target` and :class:`.Features`. For supervised case, to
    train or to score a model, use both of target and features; to predict
    only, use only the features. The length of target should be identical to
    the length of every feature of features, i.e., the number of instances.
    """
    target: Target
    """The target of the dataset."""
    features: Features
    """The features of the dataset."""

    @property
    def nrows(self):
        """Number of rows/instances."""
        if self.target.shape[0] != self.features.shape[0]:
            raise ValueError(
                "rows of target and that of features should be the same")
        return self.target.shape[0]

    @property
    def ncols(self):
        """Number of columns/features."""
        return self.features.shape[1]


class Model:
    """A base model class to be trained and to predict target based on a
    dataset.

    Before calling :meth:`.Model.train`, the model is untrained and should
    not be used to predict. After that, :meth:`.Model.predict` can be called to
    predict the :class:`.Target` of :class:`.Features`.
    """

    def train(self, dataset: Dataset) -> Model:
        """Train a model.

        Use ``self`` as a base model to train on ``dataset`` for a trained
        model.

        *Should be overridden by subclass for implementation.*

        Args:
            dataset (:class:`.Dataset`): dataset to be trained on.

        Returns:
            A trained :class:`.Model`.
        """
        raise NotImplementedError

    def predict(self, features: Features) -> Target:
        """Predict the target.

        Model returned from :meth:`.Model.train` can predict for
        :class:`.Target` on a given :class:`.Features`.

        *Should be overridden by subclass for implementation.*

        Args:
            features (:class:`.Features`): used to predicted :class:`.Target`.

        Returns:
            The prediction of :class:`.Target`.
        """
        raise NotImplementedError


class SklearnModel(Model):
    """A wrapper for ``sklearn`` models."""

    def __init__(self, model):
        """

        Args:
            model (``sklearn`` model):
        """
        self._model = model

    def train(self, dataset: Dataset) -> Model:
        """Train a model.

        Use ``self`` as a base model to train on ``dataset`` for a trained
        model. Typically the ``fit`` method of the ``sklearn`` model is used.

        Args:
            dataset (:class:`.Dataset`): dataset to be trained on.

        Returns:
            A trained :class:`.Model`.
        """
        model = self._model.fit(dataset.features, dataset.target)
        return SklearnModel(model)

    def predict(self, features: Features) -> Target:
        """Predict the target.

        Model returned from :meth:`.Model.train` can
        predict for :class:`.Target` on a given :class:`.Features`.
        Typically the ``predict`` method of the ``sklearn`` model is used.

        Args:
            features (:class:`.Features`): used to predicted :class:`.Target`.

        Returns:
            The prediction of :class:`.Target`.
        """
        return Target(self._model.predict(features))


class LoadData(NodeProcessFunction):
    """Load a dataset from an URL.

    URL can be a path. Data format can be ``SVMLIGHT``.

    Attributes:

        dformat (str): Data format. Valid options are ``SVMLIGHT``.

            .. todo::
                More format are needed

                #. CSV
                #. HDF5

        url (str): URL for the dataset. Should be a path.

            .. todo::
                More types are needed

                #. web link.
                #. sklearn data.

        target (str): The column of the dataset treated as a target.

        features (dict[int, str]):
            A mapping from 0-index of column to its name. This is useful when
            the dataset itself does not contain its own column names,
            for example, ``svmlight`` format.
    """

    def __init__(self, dformat: str, url: str, target: str,
                 features: Dict[int, str]):
        super().__init__()
        self.dformat = dformat
        self.url = url
        self.target = target
        self.features = features

    def main(self) -> Dataset:
        if self.dformat == "SVMLIGHT":
            if isinstance(sklearn_datasets, ImportError):
                raise sklearn_datasets
            features, target, *_ = sklearn_datasets.load_svmlight_file(self.url)
        else:
            raise ValueError(self.dformat)
        features = (pd.DataFrame.sparse.from_spmatrix(
            features).loc[:,
                          self.features.keys()].rename(columns=self.features))
        target = pd.Series(target, name=self.target)
        return Dataset(Target(target), Features(features))


class FilterRows(NodeProcessFunction):
    """Filter rows of :class:`.Dataset`.

    Filter rows of dataset by the value of its :class:`.Target`.

    Attributes:
        filter_by (dict[str, Any]): Indicate to filter by which
            column with what values. Typical usage is to filter ``TARGET`` with
            a list of values. For example, ``filter_by={"TARGET": [1, 2]}``
            filters the target column to only contains 1 or 2.
    """

    def __init__(self, filter_by: Dict[str, Any]):
        super().__init__()
        self.filter_by = filter_by

    def main(self, dataset: Dataset) -> Dataset:
        """

        Args:
            dataset (:class:`.Dataset`): the dataset to be filtered.

        Returns:
            A filtered dataset.
        """
        idx = dataset.target.index
        for filter_key, filter_val in self.filter_by.items():
            if filter_key == "TARGET":
                idx = idx.intersection(
                    dataset.target.index[dataset.target.isin(filter_val)])
            else:
                raise ValueError(filter_key)
        dataset.target = dataset.target.loc[idx]
        dataset.features = dataset.features.loc[idx]
        return dataset


class SamplingRows(NodeProcessFunction):
    """Sampling rows of a dataset.

    This method samples a dataset to a specific number of rows or to a ratio.

    Attributes:
        number (`int`) :
            Sampled dataset will have this many rows. Suppressed by `ratio`.

        ratio (`float`, default `1.0` if `number` is not set) :
            Sampled dataset will have `ratio * dataset.nrows` rows.
            Suppressing `number`.

        stratified (`bool`, default `False`) :
            If `True`, the dataset will be sampled using a stratified manner.
            That is, there will be same number of rows for each category of
            the dataset target, if possible.

        replace (`bool`, default `False`) :
            If `True`, the dataset will be sampled with replacement and a row
            may be selected multiple times. Will raise an exception if
            `replace` is set to `False` and `number` larger than
            `dataset.nrows` or `ratio` larger than `1`.

        random_seed (`None` or `int`, default `None`) :
            Random seed used to sample the dataset. It is used to set
            `numpy.random.BitGenerator`. See
            `Numpy Documentation
            <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.BitGenerator.html#numpy.random.BitGenerator>`_
            for more information.

        return_rest (`bool`, default `False`) :
            If `False`, only the sampled dataset is returned.

            If `True`, this method will return a dictionary of two datasets,

            .. highlight:: python
            .. code-block:: python

                {
                    'sampled': sampled_dataset,
                    'rest': rest_dataset,
                }

            `rest_dataset` contains all rows not in `sampled_dataset`.

            .. note::

                Even if `sampled_dataset` is sampled with replacement,
                `rest_dataset` does not contain duplicated rows.
    """

    def __init__(
        self,
        number: int = MISSING,
        ratio: float = MISSING,
        stratified: bool = False,
        replace: bool = False,
        random_seed: Union[None, int] = None,
        return_rest: bool = False,
    ):
        super().__init__()
        self.number = number
        self.ratio = ratio
        self.stratified = stratified
        self.replace = replace
        self.random_seed = np.random.BitGenerator(random_seed)
        self.return_rest = return_rest

        if self.ratio is MISSING and self.number is MISSING:
            self.ratio = 1.0

        self._rng = None

    def _get_number_each(self, value_counts: pd.Series, ttl_number: int):
        if self.replace or ttl_number // value_counts.shape[
                0] <= value_counts.min():
            number_each = (pd.Series(0, index=value_counts.index) +
                           ttl_number // value_counts.shape[0])
            categories = number_each.index.to_numpy(copy=True)
            self._rng.shuffle(categories)
            residual = ttl_number - number_each.sum()
            number_each += pd.Series({
                cat: 1 if i < residual else 0
                for i, cat in enumerate(categories)
            })
        else:
            number_each = pd.Series(value_counts.min(),
                                    index=value_counts.index)
            residual = self._get_number_each(
                value_counts.drop(labels=value_counts.idxmin()) -
                value_counts.min(),
                ttl_number - np.sum(number_each),
            )
            number_each += (residual +
                            pd.Series(0, index=number_each.index)).astype(int)
        return number_each

    def main(self, dataset: Dataset) -> Union[Dataset, Dict[str, Dataset]]:
        """

        Args:
            dataset (:class:`.Dataset`): the dataset to be sampled.

        Returns:
            A sampled dataset or a dictionary of the `sampled` dataset and
            the `rest` dataset.
        """
        if self.stratified:
            ttl_number = (int(self.ratio * dataset.nrows)
                          if self.ratio is not MISSING else self.number)
            value_counts = dataset.target.value_counts()
            self._rng = np.random.default_rng(self.random_seed)
            if not self.replace and ttl_number > dataset.nrows:
                raise ValueError(
                    "sampling number should at most the same size as the "
                    "dataset "
                    'when "replace" is set False')
            number_each = self._get_number_each(value_counts, ttl_number)
            selected_idx = []
            for cat, idx in dataset.target.groupby(
                    dataset.target).groups.items():
                selected_idx.append(
                    self._rng.choice(idx,
                                     number_each[cat],
                                     replace=self.replace))
            selected_idx = np.concatenate(selected_idx)
            imap = {idx: i for i, idx in enumerate(dataset.target.index)}
            selected_idx = np.array(sorted(selected_idx, key=imap.__getitem__))
            new_target = dataset.target.loc[selected_idx]
        elif self.ratio is not MISSING:
            new_target = dataset.target.sample(frac=self.ratio,
                                               replace=self.replace,
                                               random_state=self.random_seed)
        else:
            new_target = dataset.target.sample(n=self.number,
                                               replace=self.replace,
                                               random_state=self.random_seed)
        new_features = dataset.features.loc[new_target.index]
        if self.return_rest:
            rest_index = dataset.target.index.difference(new_target.index)
            rest_target = dataset.target.loc[rest_index]
            rest_features = dataset.features.loc[rest_index]
            return {
                "sampled": Dataset(Target(new_target), Features(new_features)),
                "rest": Dataset(Target(rest_target), Features(rest_features)),
            }
        return Dataset(Target(new_target), Features(new_features))


class MakeDataset(NodeProcessFunction):
    """Make a dataset using `target` and `features`."""

    def main(self, target: Target, features: Features) -> Dataset:
        """

        Args:
            target (:class:`.Target`): the input target.
            features (:class:`.Features`): the input features.

        Returns:
            A `dataset` consisted of `target` and `features`.
        """
        return Dataset(target, features)


class GetTarget(NodeProcessFunction):
    """Get `target` from a `dataset`."""

    def main(self, dataset: Dataset) -> Target:
        """

        Args:
            dataset (:class:`.Dataset`): get `target` from this `dataset`.

        Returns:
            The `target` of the `dataset`.
        """
        return dataset.target


class GetFeature(NodeProcessFunction):
    """Get `feature` from a `dataset` with a given name.

    Attributes:
        feature (str):
            This `feature` will be retrieved from the `dataset`.
    """

    def __init__(self, feature: str):
        super().__init__()
        self.feature = feature

    def main(self, dataset: Dataset) -> Feature:
        """

        Args:
            dataset (:class:`.Dataset`): get `feature` from this `dataset`.

        Returns:
            The `feature` with the given name of the `dataset`.
        """
        return Feature(dataset.features.loc[:, self.feature])


class MergeFeatures(NodeProcessFunction):
    """Merge a list of `feature` to `features`."""

    def main(self, *features: Feature) -> Features:
        """

        Args:
            *features (list of :class:`.Feature`): list of feature to be merged.

        Returns:
            A :class:`.Features` object contains the list `features`.
        """
        return Features(pd.concat([pd.Series(f) for f in features], axis=1))


def _is_number(n):
    return pd.api.types.is_number(n)


def _get_max(arr):
    if pd.api.types.is_sparse(arr):
        if arr.sparse.sp_values.shape[0] < arr.shape[0]:
            return max(arr.sparse.fill_value, np.max(arr.sparse.sp_values))
        return np.max(arr.sparse.sp_values)
    return np.max(arr)


def _get_min(arr):
    if pd.api.types.is_sparse(arr):
        if arr.sparse.sp_values.shape[0] < arr.shape[0]:
            return min(arr.sparse.fill_value, np.min(arr.sparse.sp_values))
        return np.min(arr.sparse.sp_values)
    return np.min(arr)


class LinearNormalize(NodeProcessFunction):
    """Perform linear normalization of a 1d array.

    Linearly maps the given array from range ``(u1, l1)`` to ``(u2, l2)``.

    Attributes:
        upper (`dict` of ``FROM`` and ``TO``, default both ``MAX``):
            Sets ``u1=upper["FROM"]`` and ``u2=upper["TO"]``. ``u*`` should
            be a number or a string, ``MAX`` or ``MIN``. ``MAX`` means the
            maximum of the array, and ``MIN`` means the minimum of the array.

        lower (`dict` of ``FROM`` and ``TO``, default both ``MIN``):
            Sets ``l1=lower["FROM"]`` and ``l2=lower["TO"]``. ``l*`` should
            be a number or a string, ``MAX`` or ``MIN``. ``MAX`` means the
            maximum of the array, and ``MIN`` means the minimum of the array.

    """

    def __init__(self,
                 upper: Dict[str, Any] = None,
                 lower: Dict[str, Any] = None):
        super().__init__()
        self.upper = {"FROM": "MAX", "TO": "MAX"}
        self.upper.update({} if upper is None else upper)
        self.lower = {"FROM": "MIN", "TO": "MIN"}
        self.lower.update({} if lower is None else lower)

    @staticmethod
    def _compile_limit(cmd, arr1d):
        if isinstance(cmd, str):
            if cmd == "MAX":
                return _get_max(arr1d)
            if cmd == "MIN":
                return _get_min(arr1d)
            raise RuntimeError(f"E27 {cmd}")
        if _is_number(cmd):
            return cmd
        raise ValueError(cmd)

    def main(self, arr1d: np.ndarray) -> np.ndarray:
        hifrom = self._compile_limit(self.upper.pop("FROM"), arr1d)
        hito = self._compile_limit(self.upper.pop("TO"), arr1d)
        lofrom = self._compile_limit(self.lower.pop("FROM"), arr1d)
        loto = self._compile_limit(self.lower.pop("TO"), arr1d)

        return arr1d * (hito - loto) / (hifrom - lofrom) + (
            hito * lofrom - hifrom - loto) / (hifrom - lofrom)


class MakeModel(NodeProcessFunction):
    """Make an untrained model.

    Attributes:
        method (str) : the model will use this method to train. Options are
            ``XGBOOST_REGRESSOR``, ``XGBOOST_CLASSIFIER``.
    """

    def __init__(self, method):
        super().__init__()
        self.method = method

    @staticmethod
    def _after_assign_params_routine(_from, _to):
        for k, v in list(_to.items()):
            if v is MISSING:
                del _to[k]
        warn_left_keys(_from)

    def main(self, model_params: Dict[str, Any]) -> Model:
        """

        Args:
            model_params (dict):
                The parameters to create the model. Based on the `method`,
                different parameters can be set.

                * ``XGBOOST_REGRESSOR``:

                    See more details in `XGBoost documentation
                    <https://xgboost.readthedocs.io/en/latest/parameter.html>`_

                    * ``learning_rate``
                    * ``max_depth``
                    * ``n_estimators``

                * ``XGBOOST_CLASSIFIER``:

                    See more details in `XGBoost documentation
                    <https://xgboost.readthedocs.io/en/latest/parameter.html>`_

                    * ``learning_rate``
                    * ``max_depth``
                    * ``n_estimators``

        Returns:
            An untrained :class:`.Model`.
        """
        if self.method == "XGBOOST_REGRESSOR":
            if isinstance(_xgboost, ImportError):
                raise _xgboost
            return SklearnModel(_xgboost.XGBRegressor())
        if self.method == "XGBOOST_CLASSIFIER":
            if isinstance(_xgboost, ImportError):
                raise _xgboost
            _model_params = {
                "learning_rate": model_params.pop("LEARNING_RATE", MISSING),
                "max_depth": model_params.pop("MAX_DEPTH", MISSING),
                "n_estimators": model_params.pop("N_ESTIMATORS", MISSING),
            }
            self._after_assign_params_routine(model_params, _model_params)
            return SklearnModel(
                _xgboost.XGBClassifier(**_model_params, use_label_encoder=True))
        if self.method == "SKLEARN_GRADIENT_BOOSTING_CLASSIFIER":
            if isinstance(GradientBoostingClassifier, ImportError):
                raise GradientBoostingClassifier
            _model_params = {
                "learning_rate": model_params.pop("LEARNING_RATE", MISSING),
                "max_depth": model_params.pop("MAX_DEPTH", MISSING),
                "n_estimators": model_params.pop("N_ESTIMATORS", MISSING),
            }
            self._after_assign_params_routine(model_params, _model_params)
            return SklearnModel(GradientBoostingClassifier(**_model_params))
        raise ValueError


class Train(NodeProcessFunction):
    """Train a model."""

    def main(self, model: Model, dataset: Dataset) -> Model:
        """

        Args:
            model (:class:`.Model`):
                The model to be trained.
            dataset (:class:`.Dataset`):
                The dataset to be trained on.
        Returns:
            A trained :class:`.Model`.
        """
        model = model.train(dataset)
        return model


class Predict(NodeProcessFunction):
    """Predict a target."""

    def main(self, model: Model, features: Features) -> Target:
        """

        Args:
            model (:class:`.Model`):
                The prediction is from this model.
            features (:class:`.Features`):
                The features used for prediction.
        Returns:
            A :class:`.Target` based on the `model` and `features`. The
            length of the `target` is identical to the number of rows of the
            `features`.
        """
        return model.predict(features)


class ParameterSearch(NodeProcessFunction):
    """Random search the model parameters.

    See more in :class:`utensil.random_search`.

    Attributes:

        init_state (int, default 0)
        seed (int, default 0)
        search_map (dict, default `None`)
    """

    def __init__(self, init_state=0, seed: int = 0, search_map: Dict = None):
        super().__init__()
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
                self._search_map[
                    param_name] = RandomizedParam.create_randomized_param(
                        search_type, search_option)
                self._nr_randomized_params += 1
            else:
                self._search_map[param_name] = search_method

    def _random_between(self, a, b, **kwargs):
        return self._rng.random(**kwargs) * (b - a) + a

    def _generate_seed(self):
        rand_space = []
        while True:
            base = 2**int(np.log2(self.state + 1))
            offset = self.state + 1 - base
            if offset == 0 or len(rand_space) == 0:
                linspace = np.linspace(0, 1, base + 1)
                rand_space = np.array([
                    self._random_between(
                        linspace[i],
                        linspace[i + 1],
                        size=self._nr_randomized_params,
                    ) for i in range(base)
                ])

                for i in range(self._nr_randomized_params):
                    self._rng.shuffle(rand_space[:, i])

            model_r = tuple(rand_space[offset])

            yield self.state, model_r
            self.state += 1

    def main(self):
        """

        Returns:
            Next randomly generated parameters.
        """
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
    """Calculate scores of a model, based on its prediction and a ground truth.

    Attributes:
        dataset (str):
            The name of the dataset. It is used to generate an informative
            output.
        methods (str or list of str):
             The method or a list of methods to score a model. Options are
             ``ACCURACY``.
    """

    def __init__(self,
                 dataset: str = MISSING,
                 methods: Union[str, List[str]] = None):
        super().__init__()
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
        """

        Args:

            prediction (`target`, `features` or `dataset`):
                If `prediction` is :class:`.Target`, it will be directly used
                to calculate the score without using the `model`.

                If it is :class:`.Features`, `model` will make a prediction
                based on it.

                If it is :class:`.Dataset`, `model` will make a prediction
                based on its `features`.

            ground_truth (`target` or `dataset`):
                If `ground_truth` is a :class:`.Target`, it is directly
                compared to `prediction`.

                If `ground_truth` is a :class:`.Dataset`, its `target` is
                compared to `prediction`.

            model (:class:`.Model`):
                The `model` to be scored.

                .. note::

                    If `prediction` is :class:`.Target`, then the `model` is
                    not used.

        Returns:
            A list of scoring results. A scoring result is consisted of two
            or three attributes, the scoring method name, the dataset name (
            if provided), and the score.

            For example:

            .. highlight:: python
            .. code-block:: python

                # if dataset name is 'MNIST'
                [
                    ('ACCURACY', 'MNIST', 0.812641),
                    ('FSCORE', 'MNIST', 0.713278),
                ]

                # if dataset name is not provided
                [
                    ('ACCURACY', 0.812641),
                    ('FSCORE', 0.713278),
                ]

        """
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
            return _method, self.dataset_name, _score

        ret = []
        for method in self.methods:
            if method == "ACCURACY":
                ret.append(
                    wrap_output(
                        method,
                        np.sum(prediction.values == ground_truth.values) /
                        prediction.shape[0],
                    ))
        return ret


class ChangeTypeTo(NodeProcessFunction):
    """Change the type of a given `arr`.

    Attributes:
        to_type (str):
            The `arr` will be this type. Options are ``INTEGER``, ``FLOAT``.
    """

    def __init__(self, to_type: str):
        super().__init__()
        if to_type == "INTEGER":
            self.to_type = int
        elif to_type == "FLOAT":
            self.to_type = float
        else:
            raise RuntimeError("E22")

    def main(self, arr: Union[Feature, Target]):
        """

        Args:
            arr (:class:`.Feature` or :class:`.Target`):
                The type of this will be changed.
        Returns:
            The `arr` with type changed to `to_type`.
        """
        return arr.astype(self.to_type)
