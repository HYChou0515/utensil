from __future__ import annotations

import abc
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Tuple

from utensil.general import warn_left_keys


class Parametric(abc.ABC):

    @abc.abstractmethod
    def _from_param(self, r):
        raise NotImplementedError

    def from_param(self, r):
        if not 0 <= r < 1:
            raise ValueError(f'Accept param in range [0, 1), got {r}')
        return self._from_param(r)

    @classmethod
    def create_randomized_param(cls, param_type, options) -> Parametric:
        if param_type == "BOOLEAN":
            _option = {
                "prob": options.pop("PROB"),
            }
            warn_left_keys(options)
            return BooleanParam(**_option)
        if param_type == "EXPONENTIAL_BETWEEN":
            _option = {
                "left": options.pop("LEFT"),
                "right": options.pop("RIGHT"),
                "dtype": options.pop("TYPE", float),
            }
            if _option["dtype"] == "INTEGER":
                _option["dtype"] = int
            elif _option["dtype"] == "FLOAT":
                _option["dtype"] = float
            warn_left_keys(options)
            return ExponentialBetweenParam(**_option)
        if param_type == "UNIFORM_BETWEEN":
            _option = {
                "left": options.pop("LEFT"),
                "right": options.pop("RIGHT"),
                "dtype": options.pop("TYPE", float),
            }
            if _option["dtype"] == "INTEGER":
                _option["dtype"] = int
            elif _option["dtype"] == "FLOAT":
                _option["dtype"] = float
            warn_left_keys(options)
            return UniformBetweenParam(**_option)
        if param_type == "CHOICES":
            if isinstance(options, (list, tuple)):
                if len(options) == 0:
                    raise ValueError('Should be at least one choice')
                most_common, cnt = Counter(options).most_common(1)[0]
                if cnt > 1:
                    raise ValueError('Choices should be distinct '
                                     f'(found {cnt} "{most_common}")')
                return ChoicesParam(*options)
            raise TypeError(
                f'Expect list or tuple, but got {type(options).__name__}')
        raise ValueError(f'Unsupported parametric type: "{param_type}"')


@dataclass
class BooleanParam(Parametric):
    # noinspection PyUnresolvedReferences
    """Boolean parametric.

    Attributes:
        prob: the probability of being `True`.

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)

    >>> boolean_param = BooleanParam(0.5)
    >>> positives = [boolean_param.from_param(r) for r in rng.random(1000)]
    >>> assert stats.binomtest(
    ...    sum(positives), len(positives), 0.5
    ... ).pvalue >= 0.05

    >>> boolean_param = BooleanParam(0.8)
    >>> positives = [boolean_param.from_param(r) for r in rng.random(1000)]
    >>> assert stats.binomtest(
    ...     sum(positives), len(positives), 0.8
    ... ).pvalue >= 0.05

    >>> boolean_param = BooleanParam(0.8)
    >>> boolean_param.from_param(1)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1), got 1
    """
    prob: float = 0.5

    def _from_param(self, r):
        return r < self.prob


@dataclass
class UniformBetweenParam(Parametric):
    # noinspection PyUnresolvedReferences
    """Uniform parametric between a given interval.

    Attributes:
        left: lower bound.
        right: upper bound.
        dtype: data type.

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)

    >>> param = UniformBetweenParam(-4, 2, dtype=float)
    >>> vals = [param.from_param(r) for r in rng.random(1000)]
    >>> assert stats.kstest(
    ...     vals, stats.uniform.cdf, args=(-4, 6)
    ... ).pvalue >= 0.05

    >>> param = UniformBetweenParam(-3, 10, dtype=int)
    >>> vals = [param.from_param(r) for r in rng.random(1000)]
    >>> counts = [vals.count(i) for i in range(-3, 10)]
    >>> assert sum(counts) == 1000
    >>> assert stats.chisquare(counts).pvalue >= 0.05

    dtype only support int and float
    >>> UniformBetweenParam(0, 8, dtype=dict)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict

    >>> param = UniformBetweenParam(0, 8, dtype=int)
    >>> param.dtype = dict
    >>> param.from_param(0.4)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict

    >>> param = UniformBetweenParam(0, 1, dtype=float)
    >>> param.from_param(1)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1), got 1
    """
    left: Any
    right: Any
    dtype: type

    def __post_init__(self):
        if self.dtype not in (int, float):
            raise ValueError(f'Not supporting this type: {self.dtype.__name__}')

    def _from_param(self, r):
        ret = r * (self.right - self.left) + self.left
        if self.dtype is float:
            return ret
        if self.dtype is int:
            return int(math.floor(ret))
        raise ValueError(f'Not supporting this type: {self.dtype.__name__}')


@dataclass
class ExponentialBetweenParam(Parametric):
    # noinspection PyUnresolvedReferences
    """Exponential parametric between a given interval.

    Exponential parametric is uniformly distributed in log scale.

    Attributes:
        left: lower bound
        right: upper bound
        dtype: data type

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)

    >>> param = ExponentialBetweenParam(0.01, 1024, dtype=float)
    >>> vals = [np.log(param.from_param(r)) for r in rng.random(10000)]
    >>> assert stats.kstest(
    ...     vals, stats.uniform.cdf,
    ...     args=(np.log(0.01), np.log(1024)-np.log(0.01))
    ... ).pvalue >= 0.05

    dtype only support int and float
    >>> ExponentialBetweenParam(3, 8, dtype=dict)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict

    >>> param = ExponentialBetweenParam(3, 8, dtype=int)
    >>> param.dtype = dict
    >>> param.from_param(0.4)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict

    >>> param = ExponentialBetweenParam(3, 12, dtype=float)
    >>> param.from_param(1)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1), got 1
    """
    left: Any
    right: Any
    dtype: type

    def __post_init__(self):
        if self.left <= 0 or self.right <= 0:
            raise ValueError('bounded value should be positive')
        if self.dtype not in (int, float):
            raise ValueError(f'Not supporting this type: {self.dtype.__name__}')

    def _from_param(self, r):
        log_right = math.log(self.right)
        log_left = math.log(self.left)
        ret = math.exp(r * (log_right - log_left) + log_left)
        if self.dtype is float:
            return ret
        if self.dtype is int:
            return int(math.floor(ret))
        raise ValueError(f'Not supporting this type: {self.dtype.__name__}')


@dataclass(init=False)
class ChoicesParam(Parametric):
    """Uniformly select a choice within given choices.

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> import string

    >>> param = ChoicesParam(*string.ascii_lowercase)
    >>> vals = [param.from_param(r) for r in rng.random(1000)]
    >>> counts = [vals.count(c) for c in string.ascii_lowercase]
    >>> assert sum(counts) == 1000
    >>> assert stats.chisquare(counts).pvalue >= 0.05

    >>> param.from_param(1)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1), got 1
    """
    choice: Tuple[Any]

    def __init__(self, *args: Any):
        self.choice = args

    def _from_param(self, r):
        nr_choices = len(self.choice)
        return list(self.choice)[int(r * nr_choices)]
