from __future__ import annotations

import abc
import itertools
import math
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from typing import (Any, Callable, Generator, Iterable, MutableMapping,
                    Optional, Tuple, Union)

from utensil.general import warn_left_keys


class BaseParametricSeeder(abc.ABC):
    """Parametric seeder is a parameter generator.

    It generates a tuple of values all in range [0, 1] at a time.

    Typically it is used as the input of :class:`.Parametric`.
    """

    def __init__(self, state=0, size=1, max_state=10**10, **kwargs):
        self._state = state
        self.size = size
        self.max_state = max_state

    @property
    def state(self):
        return self._state

    @abc.abstractmethod
    def _call(self) -> Generator[Tuple[float], None, None]:
        raise NotImplementedError

    def __call__(self) -> Generator[Tuple[float], None, None]:
        for rs in self._call():
            for r in rs:
                if not 0 <= r <= 1:
                    raise ValueError(
                        f'Returned param should be in range [0, 1], got {r}')
            yield rs
            self._state += 1
            if self.state >= self.max_state:
                raise ValueError(f'Reached maximum state {self.max_state}')


# noinspection PyUnresolvedReferences
class SimpleUniformParametricSeeder(BaseParametricSeeder):
    """A simple implementation of :class:`.BaseParametricSeeder`.

    >>> from scipy import stats
    >>> import numpy as np
    >>> seeds = np.array([s for s, _ in zip(SimpleUniformParametricSeeder(
    ...     rng=np.random.default_rng(0), size=3
    ... )(), range(1000))])
    >>> k1 = [stats.kstest(
    ...     seeds[:,i], stats.uniform.cdf, args=(0, 1)
    ... ).pvalue for i in range(3)]
    >>> assert min(k1) > 0.05

    When exceeding `max_iter`, `ValueError` raised.

    >>> seeder = SimpleUniformParametricSeeder(
    ...     rng=np.random.default_rng(0), max_state=2
    ... )()
    >>> next(seeder)
    (0.6369616873214543,)
    >>> next(seeder)
    (0.2697867137638703,)
    >>> next(seeder)
    Traceback (most recent call last):
    ...
    ValueError: Reached maximum state 2

    """

    def __init__(self, state=0, size=1, max_state=10**10, rng=None):
        super().__init__(state, size, max_state)
        self.rng = random if rng is None else rng

    def _call(self) -> Generator[Tuple[float], None, None]:
        while True:
            yield tuple(self.rng.random() for _ in range(self.size))


# noinspection PyUnresolvedReferences
class MoreUniformParametricSeeder(BaseParametricSeeder):
    """A more uniform parametric seeder.
    For *more* uniform, it is comparing to :class:`.SimpleParametricSeeder`.

    In pure uniform distribution, small sample size may result in *non-uniform*
    output. For example, there is `18.75%` chance for 5 uniformly distributed
    variables having range smaller than `0.5`.

    This class provide a *more uniform distributed* variables, though it is not
    actually uniform distributed. They are not i.i.d, either.

    Variables are generated via the following algorithm.

    At step 1, `x_1` is from U(0, 1).

    At step 2 and 3, `x_2`, `x_3` is from shuffle(U(0, 0.5), U(0.5, 1)).

    At step 4 to 7, `x_4` ... `x_7` is from
    shuffle(U(0, 0.25), ..., U(0.75, 1)).
    ...

    For multiple variables case, via the following algorithm.

    At step 1, `x_i1` is from U(0, 1) for every `i`.

    At step 2 and 3, `x_i2`, `x_i3` is from shuffle(U(0, 0.5), U(0.5, 1)).
    Return shuffle(`x_i2`) in step 2 and shuffle(`x_i3`) in step 3.

    At step 4 to 7, `x_i4` ... `x_i7` is from
    shuffle(U(0, 0.25), ..., U(0.75, 1)).
    Return shuffle(`x_ij`) in step j.
    ...

    >>> from scipy import stats
    >>> import numpy as np

    `seed1` and `seed2` are from `SimpleParametricSeeder`
    and `MoreUniformParametricSeeder` repectively.
    Each seeder generate 15 seeds.

    >>> seeds1 = np.array([s for s, _ in zip(SimpleUniformParametricSeeder(
    ...     rng=np.random.default_rng(0), size=1000
    ... )(), range(15))])
    >>> seeds2 = np.array([s for s, _ in zip(MoreUniformParametricSeeder(
    ...     rng=np.random.default_rng(0), size=1000
    ... )(), range(15))])

    We run a uniform stat test and calculate p-values
    of the 15 seeds for 1000 times.

    >>> k1 = [stats.kstest(
    ...     seeds1[:,i], stats.uniform.cdf, args=(0, 1)
    ... ).pvalue for i in range(1000)]
    >>> k2 = [stats.kstest(
    ...     seeds2[:,i], stats.uniform.cdf, args=(0, 1)
    ... ).pvalue for i in range(1000)]

    We should have that the max, min, median of the p-values
    of `seed2` are larger then `seed1`. So we have more confidence
    that in smaller sample size, `seed2` is more uniform than `seed1`.

    >>> assert max(k1) < max(k2)
    >>> assert min(k1) < min(k2)
    >>> assert sorted(k1)[:len(k1)//2] < sorted(k2)[:len(k2)//2]

    python random module can also be used. And all tests should be valid.

    >>> seeds3 = np.array([s for s, _ in zip(MoreUniformParametricSeeder(
    ...     rng=random, size=1000
    ... )(), range(15))])
    >>> k3 = [stats.kstest(
    ...     seeds3[:,i], stats.uniform.cdf, args=(0, 1)
    ... ).pvalue for i in range(1000)]
    >>> assert max(k1) < max(k3)
    >>> assert min(k1) < min(k3)
    >>> assert sorted(k1)[:len(k1)//2] < sorted(k3)[:len(k3)//2]

    Support :class:`numpy.random.Generator` and :mod:`random` only.

    >>> seeder = MoreUniformParametricSeeder(rng=0)
    >>> next(seeder())
    Traceback (most recent call last):
    ...
    TypeError: Non expected type of rng: int
    """

    def __init__(self, state=0, size=1, max_state=10**10, rng=None):
        super().__init__(state, size, max_state)
        self.rng = random if rng is None else rng

    def _random_between(self, a, b):
        import numpy as np
        if self.rng is random:
            return tuple(
                self.rng.random() * (b - a) + a for _ in range(self.size))
        if isinstance(self.rng, np.random.Generator):
            return self.rng.random(size=self.size) * (b - a) + a
        raise TypeError(f'Non expected type of rng: {type(self.rng).__name__}')

    def _call(self) -> Generator[Tuple[float], None, None]:
        import numpy as np
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
                    ) for i in range(base)
                ])

                for i in range(self.size):
                    self.rng.shuffle(rand_space[:, i])

            model_r = tuple(rand_space[offset])

            yield model_r


class GridParametricSeeder(BaseParametricSeeder):
    """Parametric seeder for grid search.

    This seeder generates finer and finer grids.
    For 1-dimensional case,

    #. 0, 1
    #. 1/2
    #. 1/4, 3/4
    #. 1/8, 3/8, 5/8, 7/8
    #. ...

    >>> seeder = GridParametricSeeder()
    >>> seeds = seeder()
    >>> for s, _ in zip(seeds, range(10)):
    ...     print(s)
    (0.0,)
    (1.0,)
    (0.5,)
    (0.25,)
    (0.75,)
    (0.125,)
    (0.375,)
    (0.625,)
    (0.875,)
    (0.0625,)

    For 2-dimensional case,

    #. (0, 0), (0, 1), (1, 0), (1, 1)
    #. (0, 1/2), (1/2, 0), (1/2, 1/2), (1/2, 1), (1, 1/2)
    #. ...

    >>> seeder = GridParametricSeeder(size=2)
    >>> seeds = seeder()
    >>> for s, _ in zip(seeds, range(10)):
    ...     print(s)
    (0.0, 0.0)
    (0.0, 1.0)
    (1.0, 0.0)
    (1.0, 1.0)
    (0.0, 0.5)
    (0.5, 0.0)
    (0.5, 0.5)
    (0.5, 1.0)
    (1.0, 0.5)
    (0.0, 0.25)

    A same level output can be shuffled by setting ``shuffle=True``.

    >>> seeder = GridParametricSeeder(size=2, shuffle=True, random_state=0)
    >>> seeds = seeder()
    >>> for s, _ in zip(seeds, range(10)):
    ...     print(s)
    (1.0, 0.0)
    (0.0, 0.0)
    (0.0, 1.0)
    (1.0, 1.0)
    (0.5, 0.5)
    (1.0, 0.5)
    (0.5, 0.0)
    (0.5, 1.0)
    (0.0, 0.5)
    (0.5, 0.25)

    Use `nr_points` to control the resolution of each dimension.

    >>> seeder = GridParametricSeeder(nr_points=[2, 4])
    >>> seeds = seeder()
    >>> for s, _ in zip(seeds, range(10)):
    ...     print(s)
    (0.0, 0.0)
    (0.0, 0.3333333333333333)
    (0.0, 0.6666666666666666)
    (0.0, 1.0)
    (1.0, 0.0)
    (1.0, 0.3333333333333333)
    (1.0, 0.6666666666666666)
    (1.0, 1.0)
    (0.0, 0.16666666666666666)
    (0.0, 0.5)

    `random_state` can be :class:`numpy.random.Generator`.

    >>> import numpy as np
    >>> seeder = GridParametricSeeder(size=2, shuffle=True,
    ...     random_state=np.random.default_rng(1)
    ... )
    >>> seeds = seeder()
    >>> for s, _ in zip(seeds, range(10)):
    ...     print(s)
    (0.0, 0.0)
    ...
    (0.0, 0.75)

    `random_state` can also be :mod:`random`.
    >>> import random
    >>> random.seed(0)
    >>> seeder = GridParametricSeeder(size=2, shuffle=True,
    ...     random_state=random
    ... )
    >>> seeds = seeder()
    >>> for s, _ in zip(seeds, range(10)):
    ...     print(s)
    (1.0, 0.0)
    ...
    (0.5, 0.75)

    `nr_points` should only contain values at least 2.

    >>> seeder = GridParametricSeeder(nr_points=[1])
    Traceback (most recent call last):
    ...
    ValueError: Values in nr_points should be at least 2, got 1

    `size` should be same as the length of `nr_points`.

    >>> seeder = GridParametricSeeder(nr_points=[2, 3], size=1)
    Traceback (most recent call last):
    ...
    ValueError: size=1 inconsistent to length of nr_points=2

    """

    def __init__(
        self,
        state=0,
        size=None,
        nr_points=None,
        max_state=10**10,
        shuffle=False,
        random_state=None,
    ):
        """
        Parameters:
            size(int, default to length of nr_points or 1 if it is not set):
                Size of dimensions.
            nr_points(list of int, default to 2s with length `size`):
                The initial number of points of each dimension.
                Every number should be at least 2, representing (0, 1).
                If 3, then (0, 1/2, 1).
                If 4, then (0, 1/3, 2/3, 1).
                And so on.
            shuffle(bool):
                Whether to shuffle the same level grids.
            random_state(random, numpy.random.Generator or a valid seed):
                Only matter when `shuffle` is True.
                If :mod:`random` or :class:`numpy.random.Generator`,
                it is used to do the random shuffling.
                Otherwise, it is set as the seed of
                :meth:`numpy.random.default_rng` or :mod:`random`
                if numpy cannot be imported.

        """
        if nr_points is None:
            if size is None:
                size = 1
            nr_points = [2] * size
        else:
            for nr_point in nr_points:
                if nr_point < 2:
                    raise ValueError(f'Values in nr_points '
                                     f'should be at least 2, '
                                     f'got {nr_point}')
            if size is None:
                size = len(nr_points)
            elif size != len(nr_points):
                raise ValueError(f'size={size} inconsistent to length of '
                                 f'nr_points={len(nr_points)}')
        super().__init__(state=state, size=size, max_state=max_state)
        self.nr_points = nr_points
        self.shuffle = shuffle
        if self.shuffle:
            try:
                import numpy as np
            except ImportError:
                np = None
            if np and isinstance(random_state, np.random.Generator):
                self.rng = random_state
            elif random_state is random:
                self.rng = random_state
            elif np:
                self.rng = np.random.default_rng(random_state)
            else:
                random.seed(random_state)
                self.rng = random
        else:
            self.rng = None

    @staticmethod
    def _next_grid(grid: Tuple[float]) -> Iterable[float]:
        for i in range(len(grid) * 2 - 1):
            if i % 2 == 0:
                yield grid[i // 2]
            else:
                yield (grid[i // 2] + grid[(i + 1) // 2]) / 2

    def _call(self) -> Generator[Tuple[float], None, None]:
        grids = tuple(
            (Fraction(), *(Fraction(i, nr_point - 1)
                           for i in range(1, nr_point - 1)), Fraction(1))
            for nr_point in self.nr_points)
        searched = set()

        while True:
            grid_points = itertools.product(*grids)

            if self.shuffle:
                grid_points = list(grid_points)
                self.rng.shuffle(grid_points)

            for grid_point in grid_points:
                if grid_point in searched:
                    continue
                searched.add(grid_point)
                yield tuple(float(g) for g in grid_point)

            grids = tuple(tuple(self._next_grid(grid)) for grid in grids)


class Parametric(abc.ABC):
    """Parametric is a single-variable parametric function.

    The parameter is restrict to [0, 1] to enable any parameter
    generated in this range can be used.
    Typically :class:`.BaseParametricSeeder` is intended to
    generated this kind of parameter.
    """

    @abc.abstractmethod
    def _call(self, r):
        raise NotImplementedError

    def __call__(self, r):
        if not 0 <= r <= 1:
            raise ValueError(f'Accept param in range [0, 1], got {r}')
        return self._call(r)

    @classmethod
    def create_param(cls, param_type, options) -> Parametric:
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


# noinspection PyUnresolvedReferences
@dataclass
class BooleanParam(Parametric):
    """Boolean parametric.

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)

    Should not be rejected as a binominal distribution B(N, p=0.5)

    >>> boolean_param = BooleanParam(0.5)
    >>> positives = [boolean_param(r) for r in rng.random(1000)]
    >>> assert stats.binomtest(
    ...    sum(positives), len(positives), 0.5
    ... ).pvalue >= 0.05

    Should not be rejected as a binominal distribution B(N, p=0.8)

    >>> boolean_param = BooleanParam(0.8)
    >>> positives = [boolean_param(r) for r in rng.random(1000)]
    >>> assert stats.binomtest(
    ...     sum(positives), len(positives), 0.8
    ... ).pvalue >= 0.05

    Only take input in range [0, 1]

    >>> boolean_param = BooleanParam(0.8)
    >>> boolean_param(2)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1], got 2

    Attributes:
        prob: the probability of being `True`.
    """
    prob: float = 0.5

    def _call(self, r):
        return r < self.prob


# noinspection PyUnresolvedReferences
@dataclass
class UniformBetweenParam(Parametric):
    """Uniform parametric between a given interval.

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)

    Should not be rejected as a continuous uniform distribution U(-4, 2)

    >>> param = UniformBetweenParam(-4, 2, dtype=float)
    >>> vals = [param(r) for r in rng.random(1000)]
    >>> assert stats.kstest(
    ...     vals, stats.uniform.cdf, args=(-4, 6)
    ... ).pvalue >= 0.05

    Should not be rejected as a discrete uniform distribution U(-3, 9)

    >>> param = UniformBetweenParam(-3, 10, dtype=int)
    >>> vals = [param(r) for r in rng.random(1000)]
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
    >>> param(0.4)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict

    Only take input in range [0, 1]

    >>> param = UniformBetweenParam(0, 1, dtype=float)
    >>> param(2)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1], got 2

    Attributes:
        left: lower bound.
        right: upper bound.
        dtype: data type.
    """
    left: Any
    right: Any
    dtype: type

    def __post_init__(self):
        if self.dtype not in (int, float):
            raise ValueError(f'Not supporting this type: {self.dtype.__name__}')

    def _call(self, r):
        ret = r * (self.right - self.left) + self.left
        if self.dtype is float:
            return ret
        if self.dtype is int:
            return int(math.floor(ret))
        raise ValueError(f'Not supporting this type: {self.dtype.__name__}')


# noinspection PyUnresolvedReferences
@dataclass
class ExponentialBetweenParam(Parametric):
    r"""Exponential parametric between a given interval.

    Exponential parametric is uniformly distributed in log scale.

    >>> from scipy import stats
    >>> import numpy as np
    >>> from collections import Counter
    >>> rng = np.random.default_rng(0)

    The logarithm should not be rejected as a binominal distribution.

    >>> param = ExponentialBetweenParam(0.01, 1024, dtype=float)
    >>> vals = [np.log(param(r)) for r in rng.random(10000)]
    >>> assert stats.kstest(
    ...     vals, stats.uniform.cdf,
    ...     args=(np.log(0.01), np.log(1024)-np.log(0.01))
    ... ).pvalue >= 0.05

    Also, given :math:`f` as the pdf, for all n > 0,
    :math:`\int_x^{nx} f(t)dt` should be identical.
    We can use this property to test `int` dtype.
    Here we test n=2. The sums should not be rejected as a uniform distribution.

    >>> param = ExponentialBetweenParam(1, 1024, dtype=int)
    >>> samples = [param(r) for r in np.linspace(0, 1, 10000, endpoint=False)]
    >>> cnt = Counter(samples)
    >>> cnt = [cnt[i] for i in range(1024)]
    >>> sums = [sum(cnt[i:2*i]) for i in range(1, 511)]
    >>> assert stats.chisquare(sums).pvalue >= 0.05

    The bounded values should both be positive.

    >>> param = ExponentialBetweenParam(-1, 1, dtype=float)
    Traceback (most recent call last):
    ...
    ValueError: bounded value should be positive

    dtype only support int and float

    >>> ExponentialBetweenParam(3, 8, dtype=dict)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict
    >>> param = ExponentialBetweenParam(3, 8, dtype=int)
    >>> param.dtype = dict
    >>> param(0.4)
    Traceback (most recent call last):
    ...
    ValueError: Not supporting this type: dict

    Only take input in range [0, 1]

    >>> param = ExponentialBetweenParam(3, 12, dtype=float)
    >>> param(2)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1], got 2

    Attributes:
        left: lower bound
        right: upper bound
        dtype: data type
    """
    left: Any
    right: Any
    dtype: type

    def __post_init__(self):
        if self.left <= 0 or self.right <= 0:
            raise ValueError('bounded value should be positive')
        if self.dtype not in (int, float):
            raise ValueError(f'Not supporting this type: {self.dtype.__name__}')

    def _call(self, r):
        log_right = math.log(self.right)
        log_left = math.log(self.left)
        ret = math.exp(r * (log_right - log_left) + log_left)
        if self.dtype is float:
            return ret
        if self.dtype is int:
            return int(math.floor(ret))
        raise ValueError(f'Not supporting this type: {self.dtype.__name__}')


# noinspection PyUnresolvedReferences
@dataclass(init=False)
class ChoicesParam(Parametric):
    """Uniformly select a choice within given choices.

    >>> from scipy import stats
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> import string

    Should not be rejected as a discrete uniform distribution between choices.

    >>> param = ChoicesParam(*string.ascii_lowercase)
    >>> vals = [param(r) for r in rng.random(1000)]
    >>> counts = [vals.count(c) for c in string.ascii_lowercase]
    >>> assert sum(counts) == 1000
    >>> assert stats.chisquare(counts).pvalue >= 0.05

    Only take input in range [0, 1]

    >>> param(2)
    Traceback (most recent call last):
    ...
    ValueError: Accept param in range [0, 1], got 2

    Attributes:
        choice(tuple(any)): some options to be chosen from.
    """
    choice: Tuple[Any]

    def __init__(self, *args: Any):
        self.choice = args

    def _call(self, r):
        nr_choices = len(self.choice)
        return list(self.choice)[int(r * nr_choices)]


# noinspection PyUnresolvedReferences
class SearchMap(OrderedDict, MutableMapping[str, Union[Parametric, Any]]):
    """Search map is used to direct a search method what to search.
    A search map defines a set of variables and their corresponding constraint.
    The constraint is defined by :class:`.Parametric`.
    Therefore, a search map should be a dict of str to `Parametric`.
    If an search map item has a non-`Parametric` value,
    value will be used directly.

    SearchMap inherits from :class:`collections.OrderedDict`.
    To instantiate, you can use any valid way of
    instantiating an `OrderedDict`, e.g.,

    >>> smap = SearchMap([
    ...     ('A', UniformBetweenParam(0, 1, float)),
    ...     ('B', 18),  # this is not a Parametric, will be used directly
    ...     ('C', UniformBetweenParam(10, 20, float)),
    ... ])

    `0.3` and `0.5` will be used to seeds `A` and `C`, repectively,
    because `B` is not a `Parametric`.

    >>> smap((0.3, 0.5))
    OrderedDict([('A', 0.3), ('B', 18), ('C', 15.0)])

    Number of seeds should match number of parametrics in search map.

    >>> smap((0.3, 0.5, 0.8))
    Traceback (most recent call last):
    ...
    ValueError: length of seeds (3) should be identical ... search map (2)

    Attributes:
        nr_parametric(int): number of parametrics in this map.

    """

    def __call__(
            self, seeds: Optional[Iterable[float]]
    ) -> Optional[OrderedDict[str, Any]]:
        """To generate a dict of name to parameters from a list of seeds.

        The seeds can be generated from :class:`.BaseParametricSeeder`.
        """

        if seeds is None:
            return None

        seeds = tuple(seeds)

        if len(seeds) != self.nr_parametric:
            raise ValueError(
                f'length of seeds ({len(seeds)}) should be '
                f'identical to that of search map ({self.nr_parametric})')

        seeds = iter(seeds)
        key_values = []
        for param_name, parametric in self.items():
            if isinstance(parametric, Parametric):
                key_values.append((param_name, parametric(next(seeds))))
            else:
                key_values.append((param_name, parametric))
        return OrderedDict(key_values)

    @property
    def nr_parametric(self):
        n = 0
        for parametric in self.values():
            if isinstance(parametric, Parametric):
                n += 1
        return n


class ParameterSearch(abc.ABC):
    """Search a set of parameters that maximize a function.

    `ParameterSearch` is a Monte Carlo method, where the parameter domains and
    probability distribution inside the domain is defined in
    :class:`.SearchMap`.

    The simplest usage is to use `auto_search(obj_func)`.

    The details can be found in its child classes.
    For example, :class:`.RandomSearch`.
    """

    @property
    @abc.abstractmethod
    def search_map(self) -> SearchMap:
        """the search map to generate parameters"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def parametric_seeder(self) -> BaseParametricSeeder:
        """the parametric seeder to seed the parametric in `search_map`"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ready_to_stop(self) -> bool:
        """a simple flow control indicator to stop the searching"""
        raise NotImplementedError

    @abc.abstractmethod
    def update_function_value(self, val: float):
        """to update generating rule depends on `obj_func` value.

        If the search algorithm is not iterative, this won't do anything.
        """
        raise NotImplementedError

    def __init__(self):
        self.history = []

    def auto_search(
        self,
        obj_func: Callable[..., float],
        max_iter=1,
        callbacks: Optional[Callable[[ParameterSearch], None]] = None,
        output: Optional[Union[str, Iterable[str]]] = None,
    ):
        """auto search for the parameters that maximize the objective function.
        """
        if callbacks is None:
            callbacks = []
        search = self.parameters()
        self.history = []
        for i, parameters in zip(range(max_iter), search):
            val = obj_func(**parameters)
            self.history.append((parameters, val))
            search.send(val)
            for callback in callbacks:
                callback(self)
            if self.ready_to_stop:
                break

        if output is None:
            output = ['max']
        if isinstance(output, str):
            output = [output]

        returns = []
        for o in output:
            if o == 'history':
                returns.append(self.history)
            elif o == 'max':
                returns.append(max(self.history, key=lambda x: x[1]))
            elif o == 'values':
                returns.append([pv[1] for pv in self.history])
            else:
                raise ValueError(f'Unexpected output option "{o}"')
        if len(returns) == 1:
            return returns[0]
        return returns

    def parameters(
            self) -> Generator[OrderedDict[str, Any], Optional[float], None]:
        """A generator for new parameters.

        This function is intended to be used like

        .. highlight:: python
        .. code-block:: python

            search = SomeParameterSearch().parameters()
            for parameters in search:
                val = objective_func(parameters)
                search.send(val)

                if stop_now:
                    search.send(None)
                    break

        """
        seeds = self.parametric_seeder()
        while True:
            params = self.search_map(next(seeds, None))
            if params is None:
                return
            # The pattern is next_1 -> send_1 -> next_2 -> send_2 -> ...
            # Except for next_1, in caller's perspective,
            # every next and send have to return and get something.

            # -- round 1 --
            # next_1 returns params
            # send_1 get score
            #
            # -- round 2--
            # next_2 returns params
            # send_2 get score
            #
            # ...
            score = yield params

            # -- round 1 --
            # send_1 returns nothing
            # next_2 get nothing
            #
            # -- round 2 --
            # send_2 returns nothing
            # next_3 get nothing
            #
            # ...
            yield

            self.update_function_value(score)


class RandomSearch(ParameterSearch):
    """RandomSearch is a random parameter search algorithm.

    This inherits from :class:`.ParameterSearch`,
    and uses uniform parametric seeder
    to test the objective function's value.

    To quick start, you can try

    >>> import math
    >>> def obj(x):
    ...     return (1-x)*(3-x)*(4-x)*math.log(x)

    Start searching the maximized value and its parameter.

    >>> smap = SearchMap({
    ...     'x': UniformBetweenParam(0, 6, float)
    ... })
    >>> search = RandomSearch(smap)
    >>> maximized = search.auto_search(obj, max_iter=100)
    >>> assert 3.47 <= maximized[0]['x'] <= 3.68
    >>> assert 0.759 <= maximized[1] < 0.803

    To get a deterministic output, use a random number generator with a seed.
    *The output requires using numpy.
    You may have a different output using python random module.*

    >>> search = RandomSearch(smap, random_state=0)
    >>> search.auto_search(obj, max_iter=100)
    (OrderedDict([('x', 3.5472415848224577)]), 0.7991124107876256)

    To use pure uniform distributed seeder, pass ``seeder='simple'``.

    >>> search = RandomSearch(smap, random_state=0, seeder='simple')
    >>> search.auto_search(obj, max_iter=100)
    (OrderedDict([('x', 3.5658001811981808)]), 0.8014082791616486)

    To use other seeder, pass in directly.

    >>> import numpy as np
    >>> my_seeder = MoreUniformParametricSeeder(
    ...     size=smap.nr_parametric, rng=np.random.default_rng(10)
    ... )
    >>> search = RandomSearch(smap, seeder=my_seeder)
    >>> search.auto_search(obj, max_iter=10)
    (OrderedDict([('x', 4.033554719671697)]), -0.14672480704800214)

    Set `to_stop` to `True` in a callback to stop the searching.

    >>> search = RandomSearch(smap)
    >>> def early_stop(search: RandomSearch):
    ...     if len(search.history) == 3:
    ...         search.to_stop = True
    >>> histories = search.auto_search(obj, max_iter=5,
    ...                                callbacks=[early_stop], output='history')
    >>> len(histories)
    3

    Output of `auto_search` can be adjusted by `output`.

    >>> search = RandomSearch(smap, random_state=0)
    >>> history, maximized, values = search.auto_search(
    ...     obj, max_iter=1000, output=['history', 'max', 'values']
    ... )
    >>> history
    [(OrderedDict([('x', 3.821770123928726)]), 0.5541004858153575), ...]
    >>> maximized
    (OrderedDict([('x', 3.5719289350857926)]), 0.8016437154011564)
    >>> values
    [0.5541004858153575, -0.2818357221569891, 0.26063500432307407, ...]

    Raised when non-expected output options.

    >>> search = RandomSearch(smap)
    >>> search.auto_search(obj, output='foo')
    Traceback (most recent call last):
    ...
    ValueError: Unexpected output option "foo"

    Attributes:
        search_map(SearchMap):
            the search map, see more at :class:`.SearchMap`.
        seeder(str or BaseParametricSeeder, default to 'uniform'):
            the seeder used to trigger `search_map` for input parameters.
            String options are ``uniform`` and ``simple``.
            If `uniform`, :class:`.MoreUniformParametricSeeder` will be used.
            If `simple`, :class:`.SimpleUniformParametricSeeder` will be used.
            If it is a :class:`.BaseParametricSeeder` instance,
            it will be used directly.
        random_state(None or valid random seed):
            It is the random seed for the random number generator,
            used by `seeder`.
            If numpy exists, this will be the input of
            :meth:`numpy.random.default_rng`.
            Otherwise, this will be used by :meth:`random.seed`.
        to_stop(bool):
            If set to `True`, will stop. It is intented to set in callbacks.
    """

    def __init__(self,
                 search_map: SearchMap,
                 seeder: Union[str, BaseParametricSeeder] = 'uniform',
                 random_state=None):
        super().__init__()
        self._search_map = search_map
        try:
            import numpy as np
            rng = np.random.default_rng(random_state)
        except ImportError:
            if random_state is not None:
                random.seed(random_state)
            rng = random
        if seeder == 'uniform':
            self._seeder = MoreUniformParametricSeeder(
                size=self.search_map.nr_parametric, rng=rng)
        elif seeder == 'simple':
            self._seeder = SimpleUniformParametricSeeder(
                size=self.search_map.nr_parametric, rng=rng)
        else:
            self._seeder = seeder

        self.to_stop = False

    @property
    def search_map(self) -> SearchMap:
        return self._search_map

    @property
    def parametric_seeder(self) -> BaseParametricSeeder:
        return self._seeder

    @property
    def ready_to_stop(self) -> bool:
        return self.to_stop

    def update_function_value(self, val: float):
        # for random search, there is no need to update function value
        # because it is not an iterative method.
        pass


class GridSearch(ParameterSearch):

    def __init__(self, search_map: SearchMap):
        super().__init__()
        self._search_map = search_map
