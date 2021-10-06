import os
import sys
import unittest as ut
import warnings

import pytest

from test.fixtures import FIXTURE_BASE
from utensil.loopflow.loopflow import (
    Flow,
    NodeProcessFunction,
    register_node_process_functions,
    reset_node_process_functions,
)
from utensil.loopflow.functions import basic, dataflow
from utensil.general.logger import get_logger

logger = get_logger(__name__)


class Constant(NodeProcessFunction):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self):
        return self.value


class AddValue(NodeProcessFunction):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self, a):
        return a + self.value


class Add(NodeProcessFunction):

    def __init__(self):
        super().__init__()

    def main(self, a, b):
        return a + b


class TimeValue(NodeProcessFunction):

    def __init__(self, value):
        super().__init__()
        self.value = value

    def main(self, a):
        return a * self.value


class ListAddSum(NodeProcessFunction):

    def __init__(self):
        super().__init__()

    def main(self, add, *args):
        return sum([a + add for a in args])


class Sum(NodeProcessFunction):

    def __init__(self):
        super().__init__()

    def main(self, l):
        return sum(l)


class Divide(NodeProcessFunction):

    def __init__(self):
        super().__init__()

    def main(self, a, b):
        return a / b


class Pickle(NodeProcessFunction):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def main(self, obj):
        import pickle

        pickle.dump(obj, open(self.path, "wb"))


class TestSimpleFlow(ut.TestCase):

    @pytest.mark.timeout(10)
    def test_end_to_end(self):
        reset_node_process_functions()
        register_node_process_functions(proc_func_module=sys.modules[__name__])
        register_node_process_functions(proc_funcs=[basic.GreaterThan])

        if os.path.isfile("simple.output"):
            warnings.warn("simple.output deleted")
            os.remove("simple.output")

        flow_path = os.path.join(FIXTURE_BASE, "simple.flow")
        flow = Flow.parse_yaml(flow_path)
        flow.start()

        self.assertTrue(os.path.isfile("simple.output"))
        import pickle

        output = pickle.load(open("simple.output", "rb"))
        self.assertEqual(115, output)

        os.remove("simple.output")


class TestCovtypeFlow(ut.TestCase):

    @pytest.mark.timeout(60)
    def test_end_to_end(self):
        reset_node_process_functions()
        register_node_process_functions(proc_func_module=basic)
        register_node_process_functions(proc_func_module=dataflow)

        flow_path = os.path.join(FIXTURE_BASE, "covtype.flow")
        flow = Flow.parse_yaml(flow_path)
        results = flow.start()
        logger.debug(results)
        self.assertEqual(2, len(results))
        for result in results:
            self.assertEqual("TEST_SCORE", result[0])
            self.assertEqual(1, len(result[1]))
            self.assertEqual(3, len(result[1][0]))
            self.assertEqual("ACCURACY", result[1][0][0])
            self.assertEqual("TEST_DATA", result[1][0][1])
            self.assertGreaterEqual(1, result[1][0][2])
            self.assertLessEqual(0, result[1][0][2])


if __name__ == "__main__":
    ut.main()
