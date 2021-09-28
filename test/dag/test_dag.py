import os
import unittest as ut
import warnings
from dataclasses import dataclass
from typing import Any

import pytest

from utensil.dag.dag import Dag, NodeProcessFunction
import utensil.dag.dataflow


class Constant(NodeProcessFunction):
    def __init__(self, value):
        super(self.__class__, self).__init__()
        self.value = value

    def main(self):
        return self.value


class AddValue(NodeProcessFunction):
    def __init__(self, value):
        super(self.__class__, self).__init__()
        self.value = value

    def main(self, a):
        return a + self.value


class Add(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, a, b):
        return a + b


class TimeValue(NodeProcessFunction):
    def __init__(self, value):
        super(self.__class__, self).__init__()
        self.value = value

    def main(self, a):
        return a * self.value


class ListAddSum(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, add, *args):
        return sum([a + add for a in args])


class Sum(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, l):
        return sum(l)


@dataclass
class SimpleCondition:
    c: Any
    v: Any


class LargerThan(NodeProcessFunction):
    def __init__(self, value):
        super(self.__class__, self).__init__()
        self.value = value

    def main(self, a):
        return SimpleCondition(c=a > self.value, v=a)


class Divide(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, a, b):
        return a / b


class Pickle(NodeProcessFunction):
    def __init__(self, path):
        super(self.__class__, self).__init__()
        self.path = path

    def main(self, obj):
        import pickle

        pickle.dump(obj, open(self.path, "wb"))


class TestSimpleDag(ut.TestCase):
    @pytest.mark.timeout(10)
    def test_end_to_end(self):
        if os.path.isfile("simple.output"):
            warnings.warn("simple.output deleted")
            os.remove("simple.output")

        dag_path = "simple.dag"
        dag = Dag.parse_yaml(dag_path)
        dag.start()

        self.assertTrue(os.path.isfile("simple.output"))
        import pickle

        output = pickle.load(open("simple.output", "rb"))
        self.assertEqual(115, output)

        os.remove("simple.output")


class TestCovtypeDag(ut.TestCase):
    def test_end_to_end(self):
        dag_path = "covtype.dag"
        dag = Dag.parse_yaml(dag_path)
        dag.start()


if __name__ == "__main__":
    ut.main()
