import abc
import datetime
import os
import pickle
from collections import namedtuple, OrderedDict
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from typing import Any, Union, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataclasses_json import dataclass_json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.data import AUTOTUNE


class RandomizedParam(abc.ABC):

    @abc.abstractmethod
    def from_random(self, r):
        return NotImplemented


@dataclass_json
@dataclass
class BooleanParam(RandomizedParam):

    def from_random(self, r):
        return r > 0.5


@dataclass_json
@dataclass
class UniformBetweenParam(RandomizedParam):
    left: Any
    right: Any
    otype: type

    def from_random(self, r):
        return self.otype(r * (self.right - self.left) + self.left)


@dataclass_json
@dataclass
class ExponentialBetweenParam(RandomizedParam):
    left: Any
    right: Any
    otype: type

    def from_random(self, r):
        log_right = np.log(self.right)
        log_left = np.log(self.left)
        return self.otype(np.exp(r * (log_right - log_left) + log_left))


@dataclass_json
@dataclass
class RandomizedChoices(RandomizedParam):
    choice: EnumMeta

    def from_random(self, r):
        nr_choices = len(self.choice)
        return [c for c in self.choice][-1 if r == 1 else int(r * nr_choices)]


class BatchNormChoice(Enum):
    OFF = 'off'
    AFTER_CONV2D = 'after_conv2d'
    AFTER_ACT = 'after_act'
    AFTER_POOLING = 'after_pooling'


class Conv2dPaddingChoice(Enum):
    VALID = 'valid'
    SAME = 'same'


class PoolingTypeChoice(Enum):
    MAX_POOLING = 'max'
    AVG_POOLING = 'avg'

    def get_pooling(self, **kwargs):
        if self is self.MAX_POOLING:
            return layers.MaxPooling2D(**kwargs)
        if self is self.AVG_POOLING:
            return layers.AveragePooling2D(**kwargs)
        else:
            return NotImplemented


class ActivationChoice(Enum):
    RELU = 'relu'

    def get_activation(self):
        if self is self.RELU:
            return layers.Activation('relu')
        else:
            return NotImplemented


class ModelTypeChoice(Enum):
    VGG = 'vgg'


@dataclass_json
@dataclass
class TrainConfig:
    name: str
    train_dir: str
    test_dir: str
    val_ratio: float
    train_val_seed: int
    img_height: Union[int, ExponentialBetweenParam]

    timeout: float
    neg_weight_mul: Union[float, UniformBetweenParam]
    learn_rate: Union[float, ExponentialBetweenParam]
    patience: Union[int, ExponentialBetweenParam]
    batch_size: int

    model_type: Union[Enum, RandomizedChoices]
    batch_norm_choice: Union[Enum, RandomizedChoices]
    rescaling: Union[bool, BooleanParam]
    conv2d_nr_blocks: Union[int, UniformBetweenParam]
    conv2d_first_filters: Union[int, ExponentialBetweenParam]
    conv2d_kernel_size: Union[int, UniformBetweenParam]
    conv2d_padding: Union[Enum, RandomizedChoices]
    conv2d_activation: Union[Enum, RandomizedChoices]
    pooling_type: Union[Enum, RandomizedChoices]
    pooling_size: Union[int, UniformBetweenParam]
    dense_nr_layers: Union[int, UniformBetweenParam]
    dense_units: Union[int, ExponentialBetweenParam]
    dense_activation: Union[Enum, RandomizedChoices]

    _nr_randomized_params: Union[None, int] = field(default=None)

    @property
    def img_width(self):
        return self.img_height

    @property
    def nr_randomized_params(self):
        if self._nr_randomized_params is None:
            self._nr_randomized_params = 0
            for k, v in vars(self).items():
                if isinstance(v, RandomizedParam):
                    self._nr_randomized_params += 1
        return self._nr_randomized_params

    def __post_init__(self):
        self._nr_randomized_params = None

    def randomize(self, r_list: List[float]):
        assert len(r_list) == self.nr_randomized_params
        r = iter(r_list)
        kwargs = {}
        for k, v in vars(self).items():
            if isinstance(v, RandomizedParam):
                kwargs[k] = v.from_random(next(r))
            else:
                kwargs[k] = v

        return self.__class__(**kwargs)


class Dataset:
    def __init__(self, data, labels):
        self._data = data
        self._labels = labels
        self._shape = None
        self._label_numbers = None

    def __iter__(self):
        for images, labels in self.data.take(len(self.data)):
            yield from zip(images, labels)

    def __len__(self):
        return self.shape[0]

    @classmethod
    def get_train_ds(cls, config: TrainConfig):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config.train_dir,
            validation_split=config.val_ratio,
            subset='training',
            seed=config.train_val_seed,
            image_size=(config.img_height, config.img_width),
            batch_size=config.batch_size
        )

        labels = train_ds.class_names
        return cls(train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE), labels)

    @classmethod
    def get_val_ds(cls, config: TrainConfig):
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config.train_dir,
            validation_split=config.val_ratio,
            subset='validation',
            seed=config.train_val_seed,
            image_size=(config.img_height, config.img_width),
            batch_size=config.batch_size
        )

        labels = val_ds.class_names
        return cls(val_ds.cache().prefetch(buffer_size=AUTOTUNE), labels)

    @classmethod
    def get_test_ds(cls, config: TrainConfig):
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            config.test_dir,
            image_size=(config.img_height, config.img_width),
            batch_size=config.batch_size
        )

        labels = test_ds.class_names
        return cls(test_ds.cache().prefetch(buffer_size=AUTOTUNE), labels)

    def clear_cached_properties(self):
        self._shape = None
        self._label_numbers = None

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, _labels):
        self._labels = _labels
        self.clear_cached_properties()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, _data):
        self._data = _data
        self.clear_cached_properties()

    def cal_label_numbers_and_shape(self):
        self._label_numbers = {lb: 0 for lb in self._labels}
        img_shape = None
        nr_rows = 0
        for image, label in self:
            nr_rows += 1
            if img_shape is None:
                img_shape = tuple(image.shape)
            self._label_numbers[self._labels[label.numpy()]] += 1
        self._shape = (nr_rows, *img_shape)

    def cal_shape(self):
        img_shape = None
        nr_rows = 0
        for images, labels in self.data.take(len(self.data)):
            if img_shape is None:
                if len(images) == 0:
                    img_shape = (0, 0, 0)
                else:
                    img_shape = tuple(images[0].shape)
            nr_rows += len(images)
        self._shape = (nr_rows, *img_shape)

    @property
    def shape(self):
        if self._shape is None:
            self.cal_shape()
        return self._shape

    @property
    def label_numbers(self):
        if self._label_numbers is None:
            self.cal_label_numbers_and_shape()
        return self._label_numbers

    def visualize(self, nr_row=3, nr_col=3, show_only_labels=None):
        do_show_only_labels = show_only_labels is not None

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plot_i = 0
        for image, label in self:
            if plot_i >= nr_col * nr_row:
                break
            elif do_show_only_labels and label not in show_only_labels:
                pass
            else:
                plt.subplot(nr_row, nr_col, plot_i + 1)
                plt.imshow(image.numpy().astype('uint8'))
                plt.title(self.labels[label])
                plt.axis('off')
                plot_i += 1
        plt.tight_layout()
        plt.show()


class TrainParameters:
    def __init__(self, config: TrainConfig, train_ds: Dataset, val_ds: Dataset):
        self.config = config
        self.nr_pos = train_ds.label_numbers['1']
        self.nr_neg = train_ds.label_numbers['0']


ValScore = namedtuple('ValScore', ['precision', 'recall', 'neg_loss'])
TrainScore = namedtuple('TrainScore', ['neg_loss'])
TestScore = namedtuple('TestScore', ['neg_fp', 'tp', 'neg_loss'])
SavedModel = namedtuple('SavedModel', ['weights', 'config'])


def dump_model(model: SavedModel, model_name: str):
    pickle.dump(model, open(model_name, 'wb'))


def load_model(model_name: str) -> SavedModel:
    return pickle.load(open(model_name, 'rb'))


class LearningRateScheduler2(keras.callbacks.Callback):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config: TrainConfig = config
        self.max_val_score: [ValScore, None] = None
        self.max_train_score: [TrainScore, None] = None
        self.wait = 0
        self.st_time = None

    def on_train_begin(self, logs=None):
        keras.backend.set_value(self.model.optimizer.learning_rate, self.config.learn_rate)
        print(f'training with lr={self.model.optimizer.learning_rate}')
        self.st_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        val_score = ValScore(logs[val_metric_str(MetricEnum.PRECISION)],
                             logs[val_metric_str(MetricEnum.RECALL)],
                             -logs['loss'])
        train_score = TrainScore(-logs['loss'])
        if self.max_val_score is None or self.max_val_score < val_score:
            # save the model with the best validation performance
            print(f'save weights of model with score={val_score}')
            dump_model(SavedModel(self.model.get_weights(), self.config), self.config.name)
            self.max_val_score = val_score
        if self.max_train_score is None or self.max_train_score < train_score:
            # use train performance to decide whether or not keep training
            self.max_train_score = train_score
            self.wait = 0
        else:  # bad score
            self.wait += 1
            elapsed_sec = (datetime.datetime.now() - self.st_time).total_seconds()
            to_stop = False
            if self.wait >= self.config.patience:
                print(f'waited={self.wait} timeout')
                to_stop = True
            if elapsed_sec > self.config.timeout:
                print(f'training time={(datetime.datetime.now() - self.st_time)} timeout')
                to_stop = True

            if to_stop:
                print('stop training')
                self.model.stop_training = True


class MetricEnum(Enum):
    LOSS = 'loss'
    TRUE_POSITIVES = 'tp'
    FALSE_POSITIVES = 'fp'
    TRUE_NEGATIVES = 'tn'
    FALSE_NEGATIVES = 'fn'
    PRECISION = 'pr'
    RECALL = 're'


def val_metric_str(metric_enum: MetricEnum):
    return f'val_{metric_enum.value}'


class ModelManager(abc.ABC):
    METRICS = OrderedDict((
        (MetricEnum.TRUE_POSITIVES, keras.metrics.TruePositives(name=MetricEnum.TRUE_POSITIVES.value)),
        (MetricEnum.FALSE_POSITIVES, keras.metrics.FalsePositives(name=MetricEnum.FALSE_POSITIVES.value)),
        (MetricEnum.PRECISION, keras.metrics.Precision(name=MetricEnum.PRECISION.value)),
        (MetricEnum.RECALL, keras.metrics.Recall(name=MetricEnum.RECALL.value)),
    ))

    @classmethod
    def build_model(cls, param: TrainParameters):
        for sub_class in cls.__subclasses__():
            if sub_class.model_type_name() is None:
                raise NotImplementedError
            if sub_class.model_type_name() is param.config.model_type:
                return sub_class(param)
        return NotImplemented

    @classmethod
    @abc.abstractmethod
    def model_type_name(cls) -> ModelTypeChoice:
        return NotImplemented

    def __init__(self, param: TrainParameters):
        self.param: TrainParameters = param
        self.initial_bias = np.log([param.nr_pos / param.nr_neg])

    @property
    def class_weights(self):
        neg_mul = self.param.config.neg_weight_mul / (1 + self.param.config.neg_weight_mul)
        pos_mul = 1.0 / (1 + self.param.config.neg_weight_mul)
        neg_weight = neg_mul * (self.param.nr_pos + self.param.nr_neg) / self.param.nr_neg
        pos_weight = pos_mul * (self.param.nr_pos + self.param.nr_neg) / self.param.nr_pos
        return {0: neg_weight, 1: pos_weight}

    def visualize(self):
        model = self.get_model()
        tf.keras.utils.plot_model(
            model, to_file='model.png', show_shapes=True, show_dtype=False,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(plt.imread('model.png'))
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @abc.abstractmethod
    def get_model(self) -> Sequential:
        return NotImplemented

    def fit(self, train_ds: Dataset, val_ds: Dataset, test_ds: Dataset):
        model = self.get_model()
        # model.fit(
        #     train_ds.data,
        #     validation_data=val_ds.data,
        #     callbacks=[LearningRateScheduler2(self.param.config)],
        #     class_weight=self.class_weights,
        #     epochs=10000,
        # )
        # model.set_weights(load_model(self.param.config.name).weights)
        model.evaluate(test_ds.data)  # tensorflow bug, metric of 1st evaluate may be wrong
        metrics = dict(zip((MetricEnum.LOSS, *self.METRICS.keys()), model.evaluate(test_ds.data)))
        return TestScore(
            neg_fp=-metrics[MetricEnum.FALSE_POSITIVES],
            tp=metrics[MetricEnum.TRUE_POSITIVES],
            neg_loss=-metrics[MetricEnum.LOSS]
        )


class Vgg(ModelManager):
    @classmethod
    def model_type_name(cls) -> ModelTypeChoice:
        return ModelTypeChoice.VGG

    def get_model(self):
        seq_layers = [layers.InputLayer(input_shape=(self.param.config.img_height, self.param.config.img_width, 3))]
        if self.param.config.rescaling:
            seq_layers.append(layers.experimental.preprocessing.Rescaling(1. / 255, ))

        for block_i in range(self.param.config.conv2d_nr_blocks):
            filters = self.param.config.conv2d_first_filters * self.param.config.pooling_size ** block_i
            seq_layers.append(layers.Conv2D(filters,
                                            self.param.config.conv2d_kernel_size,
                                            padding=self.param.config.conv2d_padding.value))

            if self.param.config.batch_norm_choice is BatchNormChoice.AFTER_CONV2D:
                seq_layers.append(layers.BatchNormalization())

            seq_layers.append(self.param.config.conv2d_activation.get_activation())

            if self.param.config.batch_norm_choice is BatchNormChoice.AFTER_ACT:
                seq_layers.append(layers.BatchNormalization())

            seq_layers.append(self.param.config.pooling_type.get_pooling(
                pool_size=(self.param.config.pooling_size, self.param.config.pooling_size)
            ))

            if self.param.config.batch_norm_choice is BatchNormChoice.AFTER_POOLING:
                seq_layers.append(layers.BatchNormalization())

        seq_layers.append(layers.Flatten())

        for layer_i in range(self.param.config.dense_nr_layers):
            seq_layers.append(layers.Dense(self.param.config.dense_units))
            seq_layers.append(self.param.config.conv2d_activation.get_activation())

        seq_layers.append(
            layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(self.initial_bias))
        )

        model = Sequential(seq_layers)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=list(self.METRICS.values()))
        return model


def random_between(a, b, **kwargs):
    return np.random.random(**kwargs) * (b-a) + a
    # return (np.zeros(**kwargs) + 0.5) * (b-a) + a

def main(data_home):

    train_config_template = TrainConfig(
        train_dir=os.path.join(data_home, 'LocalDataBase', 'cooked', 'OcularDiseaseRecognition', 'train'),
        test_dir=os.path.join(data_home, 'LocalDataBase', 'cooked', 'OcularDiseaseRecognition', 'test'),
        name='0',
        timeout=300,
        val_ratio=0.3,
        batch_size=64,
        train_val_seed=3,
        model_type=RandomizedChoices(ModelTypeChoice),
        img_height=64,
        learn_rate=ExponentialBetweenParam(1e-6, 1e-1, float),
        pooling_size=UniformBetweenParam(2, 5, int),
        pooling_type=RandomizedChoices(PoolingTypeChoice),
        conv2d_padding=RandomizedChoices(Conv2dPaddingChoice),
        conv2d_activation=RandomizedChoices(ActivationChoice),
        dense_activation=RandomizedChoices(ActivationChoice),
        conv2d_nr_blocks=UniformBetweenParam(2, 6, int),
        conv2d_kernel_size=UniformBetweenParam(2, 6, int),
        dense_nr_layers=UniformBetweenParam(1, 5, int),
        conv2d_first_filters=ExponentialBetweenParam(16, 128, int),
        batch_norm_choice=RandomizedChoices(BatchNormChoice),
        patience=ExponentialBetweenParam(5, 30, int),
        rescaling=BooleanParam(),
        dense_units=ExponentialBetweenParam(32, 4096, int),
        neg_weight_mul=UniformBetweenParam(0.5, 5, float),
    )

    def model_seed_gen(start):
        model_id = start
        rand_space = []
        while True:
            base = 2 ** int(np.log2(model_id+1))
            offset = model_id+1-base
            if offset == 0 or len(rand_space) == 0:
                linspace = np.linspace(0, 1, base+1)
                rand_space = np.array([random_between(
                    linspace[i], linspace[i+1],
                    size=train_config_template.nr_randomized_params
                ) for i in range(base)])

                for i in range(train_config_template.nr_randomized_params):
                    np.random.shuffle(rand_space[:, i])

            model_r = tuple(rand_space[offset])

            yield model_id, model_r
            model_id += 1

    scores_file_name = 'scores.pkl'
    try:
        scores = pickle.load(open(scores_file_name, 'rb'))
        model_seed = model_seed_gen(max(
            [int(os.path.basename(k[0])[:-6]) for k, v in sorted(scores.items(), key=lambda s: s[1])]
        )+1)
    except:
        scores = {}
        model_seed = model_seed_gen(0)

    minimum_score = TestScore(neg_fp=0, tp=0, neg_loss=0)
    first_trial = 31

    train_ds = Dataset.get_train_ds(train_config_template)
    val_ds = Dataset.get_val_ds(train_config_template)
    test_ds = Dataset.get_test_ds(train_config_template)
    while len(scores) < first_trial:
        for i in range(first_trial):
            try:
                msd = next(model_seed)
                train_config = train_config_template.randomize(msd[1])
                train_config.name = f'{msd[0]}.model'
                print(train_config)

                train_params = TrainParameters(train_config, train_ds, val_ds)
                model = ModelManager.build_model(train_params)
                score = model.fit(train_ds, val_ds, test_ds)
                print(train_config)
                print(score)
                if score > minimum_score:
                    scores[msd] = score
                    pickle.dump(scores, open(scores_file_name, 'wb'))
            except:
                pass

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
