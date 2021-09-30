from collections import namedtuple

from utensil.dag.dag import NodeProcessFunction, register_node_process_functions


class _MISSING:
    pass


MISSING = _MISSING()


class Dummy(NodeProcessFunction):
    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, a=MISSING):
        return a


class Default(NodeProcessFunction):
    def __init__(self, default):
        super(self.__class__, self).__init__()
        self.default = default

    def main(self, o=MISSING):
        if o is MISSING:
            return self.default
        return o


class Add(NodeProcessFunction):
    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, n):
        return n + self.a


ConditionValue = namedtuple("ConditionValue", ("c", "v"))


class LessEqual(NodeProcessFunction):
    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b):
        return ConditionValue(b <= self.a, b)


class Equal(NodeProcessFunction):
    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b):
        return ConditionValue(b == self.a, b)


class GreaterEqual(NodeProcessFunction):
    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b):
        return ConditionValue(b >= self.a, b)


class LessThan(NodeProcessFunction):
    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b):
        return ConditionValue(b < self.a, b)


class GreaterThan(NodeProcessFunction):
    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b):
        return ConditionValue(b > self.a, b)
