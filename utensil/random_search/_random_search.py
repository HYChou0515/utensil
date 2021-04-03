from __future__ import annotations

import abc
import hashlib
import os
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple, Dict, Union, TypeVar

from utensil.general.logger import DUMMY_LOGGER

try:
    import numpy as np
except ImportError as e:
    raise e

try:
    import pandas as pd
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


@dataclass(init=False)
class RandomizedChoices(RandomizedParam):
    choice: Tuple[Enum]

    def __init__(self, *args: Enum):
        self.choice = args

    def from_random(self, r):
        nr_choices = len(self.choice)
        return [c for c in self.choice][-1 if r == 1 else int(r * nr_choices)]


@dataclass
class RandomizedDispatcher:
    key_names: Union[str, Tuple[str]]
    dispatch: Dict[Any, RandomizedConfig]


TRandomizedConfig = TypeVar('TRandomizedConfig', bound='RandomizedConfig')


class RandomizedConfig(abc.ABC):
    def get_config(self, model_id, seed=0) -> Tuple[Dict[str, Any], TRandomizedConfig]:
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


@dataclass
class SeededConfig:
    cid: int
    base_seed: int
    seed_r: Dict[str, Any]
    config: TRandomizedConfig
    config_temp: TRandomizedConfig

    @classmethod
    def from_config_template(cls, config_temp: RandomizedConfig, model_id: int, seed: int):
        seed_r, config = config_temp.get_config(model_id=model_id, seed=seed)
        return cls(cid=model_id, base_seed=seed, seed_r=seed_r, config=config, config_temp=config_temp)


ModelScore = namedtuple('ModelScore', ['model', 'score'])


class RandomSearch(abc.ABC):
    def __init__(self, logger=None):
        self.logger = DUMMY_LOGGER if logger is None else logger

    @abc.abstractmethod
    def get_xy(self, tr_path, te_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    @abc.abstractmethod
    def model_scores_to_csv(self, model_scores) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def do_training(self, sd_config: SeededConfig, train_x, train_y, idx) -> ModelScore:
        raise NotImplementedError

    def train(self, tr_path, te_path, config_temp, model_id_range=None, seed=0):
        if model_id_range is None:
            model_id_range = range(10)

        x, y = self.get_xy(tr_path, te_path)

        te_x = x[y.isna()]
        train_x = x[~y.isna()]
        train_y = y[~y.isna()]

        rng = np.random.default_rng(seed)
        idx = np.arange(train_x.shape[0])
        rng.shuffle(idx)

        if not os.path.exists('submit'):
            os.mkdir('submit')
        model_scores = {}
        best_model = None

        for mid in model_id_range:
            self.logger.info(f'model_id={mid}: initialize')

            sd_config = SeededConfig.from_config_template(config_temp=config_temp, model_id=mid, seed=seed)
            model, score = None, None
            try:
                model, score = self.do_training(sd_config, train_x, train_y, idx)
            except Exception:
                self.logger.warning(f'model_id={sd_config.cid}: invalid config for model_id={sd_config.cid}, {seed=}',
                               stack_info=True)

            self.logger.info(f'model_id={sd_config.cid}: {score=}')

            # record model_id, model_config and score
            model_scores[sd_config.cid] = sd_config.config.to_plain_dict()
            assert 'score' not in model_scores[sd_config.cid]
            model_scores[sd_config.cid]['score'] = score
            self.model_scores_to_csv(model_scores)

            # keep the best model
            if score is not None and (best_model is None or best_model[0] < score):
                best_model = (score, deepcopy(model), sd_config)

                # use the current best model to generate csv for submission
                te_xpy = np.empty(shape=(te_x.shape[0], 2), dtype=int)
                te_xpy[:, 0] = np.arange(te_x.shape[0]) + 1
                te_xpy[:, 1] = best_model[1].predict(te_x)
                submit_path = os.path.join('submit', f'{__name__}_{int(np.round(score * 1e5))}_{sd_config.cid:04d}.csv')
                pd.DataFrame(te_xpy, columns=['ImageId', 'Label']).to_csv(submit_path, index=False)
            if best_model is not None:
                self.logger.info(
                    f'model_id={sd_config.cid}: current best model_id={best_model[2].cid}, score={best_model[0]}, '
                    f'config={best_model[2].config.to_plain_dict()}')

        return best_model, model_scores
