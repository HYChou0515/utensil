import os
import unittest as ut
from typing import Generator, Tuple, Dict
from unittest.mock import patch

import pytest

from utensil.param_search import parametric

LOOPFLOW_INSTALLED = os.environ.get('LOOPFLOW_INSTALLED', '1') == '1'


class TestBaseParametricSeeder(ut.TestCase):

    # pylint: disable=abstract-class-instantiated
    @patch.object(parametric.BaseParametricSeeder, '__abstractmethods__', set())
    def test_cannot_be_called_directly(self):
        base_seeder = parametric.BaseParametricSeeder()
        with pytest.raises(NotImplementedError):
            next(base_seeder())

    def test_valid_inherits(self):

        class BadParametricSeeder(parametric.BaseParametricSeeder):

            def _call(self) -> Generator[Tuple[float], None, None]:
                yield 1, 2, 3

        bad_seeder = BadParametricSeeder()
        with pytest.raises(
                ValueError,
                match=r'Returned param should be in range \[0, 1\], got 2'):
            next(bad_seeder())

    def test_state_larger_than_number_of_seeds(self):

        class ShortSeeder(parametric.BaseParametricSeeder):

            def _call(self) -> Generator[Tuple[float], None, None]:
                yield 0.1, 0.2, 0.3

        seeder = ShortSeeder(state=5)
        self.assertTrue(next(seeder(), None) is None)


class TestParametric(ut.TestCase):

    # pylint: disable=abstract-class-instantiated
    @patch.object(parametric.Parametric, '__abstractmethods__', set())
    def test_cannot_be_called_directly(self):
        base_parametric = parametric.Parametric()
        with pytest.raises(NotImplementedError):
            base_parametric(0)


@pytest.mark.parametrize('redundant_var', [True, False])
@pytest.mark.parametrize('param_type,options', [
    ('BOOLEAN', {
        'PROB': 0.3
    }),
    ('EXPONENTIAL_BETWEEN', {
        'LEFT': 0.1,
        'RIGHT': 10
    }),
    ('EXPONENTIAL_BETWEEN', {
        'LEFT': 0.1,
        'RIGHT': 10,
        'TYPE': 'FLOAT'
    }),
    ('EXPONENTIAL_BETWEEN', {
        'LEFT': 0.1,
        'RIGHT': 10,
        'TYPE': 'INTEGER'
    }),
    ('UNIFORM_BETWEEN', {
        'LEFT': 0.1,
        'RIGHT': 10
    }),
    ('UNIFORM_BETWEEN', {
        'LEFT': 0.1,
        'RIGHT': 10,
        'TYPE': 'FLOAT'
    }),
    ('UNIFORM_BETWEEN', {
        'LEFT': 0.1,
        'RIGHT': 10,
        'TYPE': 'INTEGER'
    }),
])
def test_parametric_create_param(param_type, options: Dict, redundant_var):
    options = {**options}
    if redundant_var:
        options['FOO'] = 'BAZ'
    with pytest.warns(SyntaxWarning if redundant_var else None) as record:
        param = parametric.Parametric.create_param(param_type, options)
        assert isinstance(param, parametric.Parametric)
    if redundant_var:
        assert len(record) == 1
    else:
        assert not record, "No warnings should have been raised"


@pytest.mark.parametrize('param_type,options', [
    ('CHOICES', ['A']),
    ('CHOICES', ('A',)),
    ('CHOICES', ('A', 'B')),
])
def test_parametric_choices(param_type, options):
    param = parametric.Parametric.create_param(param_type, options)
    assert isinstance(param, parametric.Parametric)


def test_parametric_choices_should_be_distinct():
    with pytest.raises(ValueError,
                       match=r'Choices should be distinct \(found 3 "A"\)'):
        parametric.Parametric.create_param('CHOICES',
                                           ['A', 'A', 'A', 'B', 'B', 'C'])


def test_parametric_choices_should_not_be_empty():
    with pytest.raises(ValueError, match='Should be at least one choice'):
        parametric.Parametric.create_param('CHOICES', [])


def test_parametric_choices_should_be_list_or_tuple():
    with pytest.raises(TypeError, match='Expect list or tuple, but got dict'):
        parametric.Parametric.create_param('CHOICES', {})


def test_parametric_unexpected_param_type():
    with pytest.raises(ValueError, match='Unsupported parametric type: "FOO"'):
        parametric.Parametric.create_param('FOO', {})


class TestParameterSearch(ut.TestCase):

    # pylint: disable=pointless-statement
    # pylint: disable=abstract-class-instantiated
    @patch.object(parametric.ParameterSearch, '__abstractmethods__', set())
    def test_cannot_be_called_directly(self):
        base_param_search = parametric.ParameterSearch()
        with pytest.raises(NotImplementedError):
            base_param_search.search_map
        with pytest.raises(NotImplementedError):
            base_param_search.parametric_seeder
        with pytest.raises(NotImplementedError):
            base_param_search.ready_to_stop
        with pytest.raises(NotImplementedError):
            base_param_search.update_function_value(0)


class TestRandomSearch(ut.TestCase):

    @pytest.mark.skipif(condition=LOOPFLOW_INSTALLED,
                        reason="test behavior when loopflow not installed")
    def test_random_search_without_numpy(self):
        from collections import OrderedDict
        import math

        def obj(x):
            return (1 - x) * (3 - x) * (4 - x) * math.log(x)

        smap = parametric.SearchMap(
            {'x': parametric.UniformBetweenParam(0, 6, float)})
        search = parametric.RandomSearch(smap, random_state=0, seeder='simple')
        maximized = search.auto_search(obj, max_iter=100)
        assert maximized == (OrderedDict([('x', 3.5777211694986377)]),
                             0.8016234110091472)


class TestGridSearchSeeder(ut.TestCase):

    @pytest.mark.skipif(condition=LOOPFLOW_INSTALLED,
                        reason="test behavior when loopflow not installed")
    def test_grid_search_seeder_without_numpy(self):
        seeder = parametric.GridParametricSeeder(size=2,
                                                 shuffle=True,
                                                 random_state=0)
        expects = [
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.5),
            (0.5, 1.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 0.5),
            (0.5, 0.75),
        ]
        for seeds, expected in zip(seeder(), expects):
            self.assertEqual(expected, seeds)


def test_short_seeder_for_search():

    class ShortSeeder(parametric.BaseParametricSeeder):

        def _call(self) -> Generator[Tuple[float], None, None]:
            yield from [(1,), (1 / 2,), (0,)]

    import math

    def obj(x):
        return (1 - x) * (3 - x) * (4 - x) * math.log(x)

    smap = parametric.SearchMap(
        {'x': parametric.UniformBetweenParam(1, 6, float)})
    search = parametric.RandomSearch(smap, seeder=ShortSeeder())
    history = search.auto_search(obj, max_iter=100, output='history')
    assert len(history) == 3
