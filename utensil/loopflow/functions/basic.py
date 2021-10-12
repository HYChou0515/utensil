"""**Provide NodeProcessFunction for basic usage.**

Example:

.. highlight:: python
.. code-block:: python

    from utensil.loopflow.functions import basic
    from utensil.loopflow.loopflow import register_node_process_functions
    register_node_process_functions(basic)
"""
from collections import namedtuple
from typing import Any

from utensil.loopflow.loopflow import NodeProcessFunction


class _MISSING:
    pass


MISSING = _MISSING()
"""Missing token.

Used to indicate a missing value.
"""


class Dummy(NodeProcessFunction):
    """Identical function.

    Returns whatever it get.

    >>> Dummy().main('anything')
    'anything'
    """

    def main(self, a: Any = MISSING):
        return a


class Default(NodeProcessFunction):
    """Implements a default behavior.

    Return a default value if triggered before getting anything.

    Attributes:
        default: the default value.

    >>> default = Default('my_default')

    This will return the input.
    >>> default.main('my_input')
    'my_input'

    This will return the default value.
    >>> default.main()
    'my_default'
    """

    def __init__(self, default):
        super().__init__()
        self.default = default

    def main(self, o: Any = MISSING):
        if o is MISSING:
            return self.default
        return o


class Add(NodeProcessFunction):
    """Add a predefined constant, i.e., ``n+a``.

    Attributes:
        a: the constant value to be added.

    >>> p = Add(3)
    >>> p.main(5)
    8
    >>> p.main(9)
    12
    """

    def __init__(self, a):
        super().__init__()
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

    >>> LessEqual(3).main(3)
    ConditionValue(c=True, v=3)
    >>> LessEqual(5).main(10)
    ConditionValue(c=False, v=10)
    """

    def __init__(self, a):
        super().__init__()
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

    >>> Equal(3).main(3)
    ConditionValue(c=True, v=3)
    >>> Equal(5).main(10)
    ConditionValue(c=False, v=10)
    """

    def __init__(self, a):
        super().__init__()
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

    >>> GreaterEqual(3).main(3)
    ConditionValue(c=True, v=3)
    >>> GreaterEqual(15).main(10)
    ConditionValue(c=False, v=10)
    """

    def __init__(self, a):
        super().__init__()
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

    >>> LessThan(3).main(3)
    ConditionValue(c=False, v=3)
    >>> LessThan(15).main(10)
    ConditionValue(c=True, v=10)
    """

    def __init__(self, a):
        super().__init__()
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

    >>> GreaterThan(3).main(3)
    ConditionValue(c=False, v=3)
    >>> GreaterThan(5).main(10)
    ConditionValue(c=True, v=10)
    """

    def __init__(self, a):
        super().__init__()
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