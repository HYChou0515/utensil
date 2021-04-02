from __future__ import annotations

import abc
import hashlib
from dataclasses import dataclass
from enum import EnumMeta
from typing import Any, Tuple, Dict, Union, TypeVar

try:
    import numpy as np
except ImportError as e:
    raise e


class RandomizedParam(abc.ABC):
    @abc.abstractmethod
    def from_random(self, r):
        raise NotImplemented


@dataclass
class BooleanParam(RandomizedParam):
    def from_random(self, r):
        return r > 0.5


@dataclass
class UniformBetweenParam(RandomizedParam):
    left: Any
    right: Any
    otype: type

    def from_random(self, r):
        return self.otype(r * (self.right - self.left) + self.left)


@dataclass
class ExponentialBetweenParam(RandomizedParam):
    left: Any
    right: Any
    otype: type

    def from_random(self, r):
        log_right = np.log(self.right)
        log_left = np.log(self.left)
        return self.otype(np.exp(r * (log_right - log_left) + log_left))


@dataclass
class RandomizedChoices(RandomizedParam):
    choice: EnumMeta

    def from_random(self, r):
        nr_choices = len(self.choice)
        return [c for c in self.choice][-1 if r == 1 else int(r * nr_choices)]


@dataclass
class RandomizedDispatcher:
    key_names: Union[str, Tuple[str]]
    dispatch: Dict[Any, RandomizedConfig]


T = TypeVar('T', bound='RandomizedConfig')


class RandomizedConfig(abc.ABC):
    def get_config(self, model_id, seed=0) -> Tuple[Any, T]:
        kwargs = {}
        dispatchers = {}
        params = {}
        model_r = {}
        for k, v in vars(self).items():
            if isinstance(v, RandomizedParam):
                params[k] = v
            elif isinstance(v, RandomizedDispatcher):
                dispatchers[k] = v
                kwargs[k] = v
            else:
                kwargs[k] = v
                model_r[k] = None

        base = 2 ** int(np.log2(model_id + 1))
        offset = model_id + 1 - base
        sd = int.from_bytes(hashlib.sha256(str(seed + base).encode()).digest()[:4], 'big')
        rng = np.random.default_rng(sd)
        linspace = np.linspace(0, 1, base + 1)
        rand_space = rng.random(size=(base, len(params))) * (linspace[1] - linspace[0]) + linspace[:-1].reshape(-1, 1)

        for i in range(len(params)):
            rng.shuffle(rand_space[:, i])

        for (k, v), r in zip(params.items(), rand_space[offset]):
            model_r[k] = r
            kwargs[k] = v.from_random(r)

        model_c = self.__class__(**kwargs)
        for var_name, dispatcher in dispatchers.items():
            if isinstance(dispatcher.key_names, str):
                key = vars(model_c)[dispatcher.key_names]
            else:
                key = tuple(vars(model_c)[kn] for kn in dispatcher.key_names)
            r, vars(model_c)[var_name] = dispatcher.dispatch[key].get_config(model_id, seed=seed)
            model_r[var_name] = r
        return model_r, model_c

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in vars(self).items():
            if isinstance(v, RandomizedConfig):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def to_plain_dict(self, sep=':') -> Dict[str, Any]:
        d = {}
        for k, v in vars(self).items():
            if isinstance(v, RandomizedConfig):
                for vk, vv in v.to_plain_dict().items():
                    d[f'{k}{sep}{vk}'] = vv
            else:
                d[k] = v
        return d
