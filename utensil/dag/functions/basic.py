"""**Provide NodeProcessFunction for basic usage.**

Example:

.. highlight:: python
.. code-block:: python

    from utensil.dag.functions import basic
    from utensil.dag.dag import register_node_process_functions
    register_node_process_functions(basic)
"""
from collections import namedtuple

from utensil.dag.dag import NodeProcessFunction


class _MISSING:
    pass


MISSING = _MISSING()
"""Missing token.

Used to indicate a missing value.
"""


class Dummy(NodeProcessFunction):
    """Identical function.

    Returns whatever it get.
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def main(self, a=MISSING):
        return a


class Default(NodeProcessFunction):
    """Implements a default behavior.

    Return a default value if triggered before getting anything.

    Attributes:
        default: the default value.
    """

    def __init__(self, default):
        super(self.__class__, self).__init__()
        self.default = default

    def main(self, o=MISSING):
        if o is MISSING:
            return self.default
        return o


class Add(NodeProcessFunction):
    """Add a predefined constant, i.e., ``n+a``.

    Attributes:
        a: the constant value to be added.
    """

    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, n):
        """

        Args:
            n: value to be added with ``a``.

        Returns:
            ``n+a``.
        """
        return n + self.a


ConditionValue = namedtuple("ConditionValue", ("c", "v"))
"""A pair of a boolean and a value for flow control.

Attributes:
    c: a boolean value indicating if condition is passed.
    v: the value to be used.
"""


class LessEqual(NodeProcessFunction):
    """Check is less than or equal to a constant, i.e., ``b <= a``.

    Attributes:
        a: the constant value to be compared with.
    """

    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b) -> ConditionValue:
        """

        Args:
            b: value to be compared with ``a``.

        Returns:
            a :class:`.ConditionValue`, with ``c`` is True if ``b <= a``,
            and ``v`` is ``b``.
        """
        return ConditionValue(b <= self.a, b)


class Equal(NodeProcessFunction):
    """Check is equal to a constant, i.e., ``b == a``.

    Attributes:
        a: the constant value to be compared with.
    """

    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b) -> ConditionValue:
        """

        Args:
            b: value to be compared with ``a``.

        Returns:
            a :class:`.ConditionValue`, with ``c`` is True if ``b == a``,
            and ``v`` is ``b``.
        """
        return ConditionValue(b == self.a, b)


class GreaterEqual(NodeProcessFunction):
    """Check is greater than or equal to a constant, i.e., ``b >= a``.

    Attributes:
        a: the constant value to be compared with.
    """

    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b) -> ConditionValue:
        """

        Args:
            b: value to be compared with ``a``.

        Returns:
            a :class:`.ConditionValue`, with ``c`` is True if ``b >= a``,
            and ``v`` is ``b``.
        """
        return ConditionValue(b >= self.a, b)


class LessThan(NodeProcessFunction):
    """Check is less than a constant, i.e., ``b < a``.

    Attributes:
        a: the constant value to be compared with.
    """

    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b) -> ConditionValue:
        """

        Args:
            b: value to be compared with ``a``.

        Returns:
            a :class:`.ConditionValue`, with ``c`` is True if ``b < a``,
            and ``v`` is ``b``.
        """
        return ConditionValue(b < self.a, b)


class GreaterThan(NodeProcessFunction):
    """Check is greater than a constant, i.e., ``b > a``.

    Attributes:
        a: the constant value to be compared with.
    """

    def __init__(self, a):
        super(self.__class__, self).__init__()
        self.a = a

    def main(self, b) -> ConditionValue:
        """

        Args:
            b: value to be compared with ``a``.

        Returns:
            a :class:`.ConditionValue`, with ``c`` is True if ``b > a``,
            and ``v`` is ``b``.
        """
        return ConditionValue(b > self.a, b)
