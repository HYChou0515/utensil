from __future__ import annotations

import abc
import itertools as it
import re
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue, SimpleQueue
from queue import Empty
from types import ModuleType
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, Union

from utensil import get_logger
from utensil.general import open_utf8, warn_left_keys

logger = get_logger(__name__)


class _SyntaxToken(str, Enum):
    """Syntax token used in flow definition."""

    # Flow tokens
    FLOW = "FLOW"
    """
    Used to define a flow object.
    """
    # Node tokens
    SENDERS = "SENDERS"
    """
    Used to define a senders object.
    """
    CALLERS = "CALLERS"
    """
    Used to define a callers object.
    """
    TASK = "TASK"
    """
    Used to define a task object.
    """
    EXPORT = "EXPORT"
    """
    Used to define how to export.
    """
    END = "END"
    """
    Used to define whether a node is an end of flow.
    """
    # Special tokens
    SWITCHON = "SWITCHON"
    """
    A special token to indicate a starting point.
    """


_S = _SyntaxToken
"""Just an abbreviation for local usage."""


class _ExportOperator(str, Enum):
    """Identifier for export operator.

    PRINT: print out node status after finishing tasks

    RETURN: put return value from tasks as a flow return value

    """

    PRINT = "PRINT"
    RETURN = "RETURN"


class TriggerToken:
    """A token for a Caller to trigger its Callee.

    Because the return value of a Caller is only used for checking
    the constraints, we don't need to pass it to its Callee.
    Instead, `TriggerToken` is passed.
    """


class EndOfFlowToken:
    """A token to broadcast the end-of-flow event to all nodes.

    It is put into the end queue to notify all nodes to stop their
    workers and themselves.
    Most of the cases, EndOfFlowToken is put by a Node labeled end-of-flow
    or a NodeWorker counters an unexpected exception.
    """


class BaseNode(Process):
    """A base class for Node"""


class BaseNodeWorker(Process):
    """A base class for NodeWorker"""


class NodeTask(abc.ABC):
    """Task for a node to execute.

    Essentially, it is a configurable function.
    `__init__` is used for configuration, and a node is executing
    the `__call__`.

    A derived class should at least implement the `main` and a node is using
    `__call__`, a wrapper of `main`.

    Parameters passed from the node's parents are passed to `main` of the
    first task of the node. The second task's `main` is taking the first one's
    result, and so on.

    A derived class can override the `__init__` to define how this task
    can be configured.

    Parsing
    =======
    To parse an object to a executable `NodeTask`,
    a derived class must be defined and registered in the global task map.

    As a subclass of `NodeTask`, Mytask is defined as

    >>> class MyTask(NodeTask):
    ...     def __init__(self, a=3, b=5, c=10):
    ...         self.a = a
    ...         self.b = b
    ...         self.c = c
    ...     def main(self):
    ...         return self.a, self.b, self.c

    and registered.

    >>> reset_node_tasks()
    >>> register_node_tasks(tasks=[MyTask])

    To parse an object to a `NodeTask`, the object must be
    one of the following types:

    1. `str`

    >>> obj = 'MY_TASK'
    >>> tasks = NodeTask.parse(obj)
    >>> tasks[0]()
    (3, 5, 10)

    2. `dict` with exactly one item, where the item is of type

        1. `list`

        >>> obj = {'MY_TASK': ['FOO', 'BAR']}
        >>> tasks = NodeTask.parse(obj)
        >>> tasks[0]()
        ('FOO', 'BAR', 10)

        2. `dict`

        >>> obj = {'MY_TASK': {'a': 'FOO', 'c': 'BAR'}}
        >>> tasks = NodeTask.parse(obj)
        >>> tasks[0]()
        ('FOO', 5, 'BAR')

        3. `str`

        >>> obj = {'MY_TASK': 'FOO'}
        >>> tasks = NodeTask.parse(obj)
        >>> tasks[0]()
        ('FOO', 5, 10)

    `NodeTask.parse` is trying to parse a list of `NodeTask`,
    so it also accepts a list of valid object described above,
    and returns a list of `NodeTask`.
    This list of objects can specify different `NodeTask`.

    >>> class SimpleTask(NodeTask):
    ...     def main(self):
    ...         return 'faz'
    >>> register_node_tasks(tasks=[SimpleTask])
    >>> obj = [{'MY_TASK': {'b': 'foo'}}, 'SIMPLE_TASK']
    >>> tasks = NodeTask.parse(obj)
    >>> tasks[0]()
    (3, 'foo', 10)
    >>> tasks[1]()
    'faz'

    """

    def __init__(self, *args, **kwargs):
        """CAN be overridden for task configuration."""

    @abstractmethod
    def main(self, *args, **kwargs):
        """MUST be overridden by its derived class."""
        raise NotImplementedError

    @classmethod
    def parse(cls, o) -> List[NodeTask]:
        """Parse a list of valid object for a list of NodeTask.

        If the object is not a list, it is wrapped in a list and then
        be parsed. So the result is still a list of NodeTask.
        """
        task_map = _NODE_TASK_MAPPING

        def _parse_1(_o):
            if isinstance(_o, str):
                return task_map[_o]()
            if isinstance(_o, dict):
                if len(_o) != 1:
                    raise SyntaxError(
                        "Task specified by a dict should have exactly"
                        f" a pair of key and value, got {_o}")
                name, params = _o.popitem()
                logger.debug(name, params)
                if isinstance(params, list):
                    return task_map[name](*params)  # noqa
                if isinstance(params, dict):
                    params = {k.lower(): v for k, v in params.items()}
                    return task_map[name](**params)  # noqa
                return task_map[name](params)  # noqa
            raise SyntaxError(
                f"Task item support parsed by only str and dict, got {_o}")

        if not isinstance(o, list):
            o = [o]

        return [_parse_1(_) for _ in o]

    def __call__(self, *args, **kwargs):
        """A wrapper of `main` to be called by a node worker."""

        # use the required parameters only.
        params = [
            kwargs.pop(k) for k in self.main.__code__.co_varnames if k in kwargs
        ]
        return self.main(*params, *args, **kwargs)


@dataclass
class NodeMeta:
    """Shared information between a `Node` and a `NodeWorker`."""

    node_name: str
    callees: List[Node]
    receivers: List[Node]
    tasks: List[NodeTask]
    export: Tuple[str]
    result_q: SimpleQueue
    end_q: SimpleQueue


class NodeWorkerBuilder:
    """For a `Node` to build up a `NodeWorker`.

    A `NodeMeta` should be defined before building a worker.

    >>> NodeWorkerBuilder().build()
    Traceback (most recent call last):
    ...
    RuntimeError: meta should be defined before build a node worker

    """

    def __init__(self):
        self.meta: Optional[NodeMeta] = None

    def build(self, *args, **kwargs):
        if self.meta is None:
            raise RuntimeError(
                "meta should be defined before build a node worker")
        kwargs = {k.lower(): v for k, v in kwargs.items()}
        worker = NodeWorker(self.meta, *args, **kwargs)
        return worker


class NodeWorker(BaseNodeWorker):
    """A worker started by a node for executing some specific tasks.

    When a node is ready (either its senders or callers are ready),
    it starts a worker, with parameters from its senders, to execute
    some specific tasks. The worker has a shorter life time than a `Node`,
    meaning it is terminated after finished the tasks.

    The parameters from its senders will be passed to the first task.
    The output of the first task is passed to the second task, and so on.

    If a node is marked with export PRINT, the node's name and its final
    return value (the output of the last task) is printed.
    If a node is marked with export RETURN, the final return value is
    put into the result queue, as the final result of the flow.

    A worker is also responsible for calling the node's callees and
    send its final return value to the node's receivers.

    >>> import multiprocessing
    >>> class MyTask(NodeTask):
    ...     def __init__(self, a=3, b=5, c=10):
    ...         self.a = a
    ...         self.b = b
    ...         self.c = c
    ...     def main(self, m, n=1):
    ...         return m*self.a + n*self.b + self.c
    >>> end_q = multiprocessing.SimpleQueue()
    >>> result_q = multiprocessing.SimpleQueue()
    >>> meta = NodeMeta("foo", [], [],
    ...                 [MyTask(5), MyTask(), MyTask(10, 20, 30)],
    ...                 ("PRINT", "RETURN"), result_q, end_q)
    >>> worker = NodeWorker(meta, 7, 13)
    >>> worker.run()  # use worker.start() to start a new process in real case
    foo: 3500
    >>> result_q.get()
    ('foo', 3500)
    >>> end_q.empty()
    True
    """

    def __init__(self, meta: NodeMeta, *args, **kwargs):
        super().__init__()
        self.meta: NodeMeta = meta
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        logger.debug("Node {} started", self.meta.node_name)
        try:
            ret = self.meta.tasks[0](*self.args, **self.kwargs)
            for task in self.meta.tasks[1:]:
                ret = task(ret)
            if _ExportOperator.PRINT in self.meta.export:
                print(f"{self.meta.node_name}: {ret}")
            if _ExportOperator.RETURN in self.meta.export:
                self.meta.result_q.put((self.meta.node_name, ret))
            # MUST send before trigger.
            # Receivers wait for callers, but callees do not wait for senders.
            # That is, it might happen that receiver got triggered before
            # sender ready.
            # For example, A triggers B, B triggers C, and A send C.
            # If trigger before send, after A ready,
            # C might get triggered by B (triggered by A) before A send C.
            for receiver in self.meta.receivers:
                receiver.receive(ret, self.meta.node_name)
            for callee in self.meta.callees:
                callee.triggered_by(ret, self.meta.node_name)
        except Exception as err:
            logger.exception(err)
            self.meta.end_q.put(EndOfFlowToken())
            raise err


class _Operator(str, Enum):
    """Operators for `ParentSpecifierToken`."""

    OR = "|"
    FLOW_IF = "/"
    FLOW_OR = ","
    SUB = "."
    REGEX_SUB = r"\."
    FLOW_USE = "="


@dataclass(frozen=True)
class ParentSpecifierToken:
    r"""An atomic element to specify how and when a parent should be used.

    The `how` means which attribute of the parent node is passed to its child.
    The `when` means under what condition the parent node is passing its
    output to its child.
    The `atomic` means it is the minimal building block for a full
    specification defined a node's several parents should be used.

    Note that to make a valid ParentSpecifierToken to work on attributes,
    the attributes should only be lower cases.

    In the following example, 'MY_TASK.BAR' can be retreived,
    'MY_TASK.BAZ' cannot.

    .. highlight:: python
    .. code-block:: python

        class Foo:
            def __init__(self):
                self.bar = 3  # do this
                self.Baz = 5  # don't do this

        class MyTask(NodeTask):
            def main(self):
                return Foo()

    A token accpets the following regular expression.

    >>> print(r'\w+(\.\w+)*(/\w+(,\w+)*)?(=\w+(\.\w+)*)?')
    \w+(\.\w+)*(/\w+(,\w+)*)?(=\w+(\.\w+)*)?

    A simplest token looks like

    >>> s = 'FOO'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.node_name
    'FOO'

    Retrieving a specific attribute from the ouptut of a node

    >>> s = 'FOO.BAR'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.attrs_to_return
    ('BAR',)

    Flow to its child only if satisfying a specific condition.
    E.g., only if the output of FOO equals to BAZ,
    its child gets the output of FOO.

    >>> s = 'FOO/BAZ'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.flow_constraints
    ('BAZ',)

    To use a specific attribute to check if it satisfies a
    specific condition.
    E.g., only if BAR in the output of FOO equals to BAZ,
    its child gets the output of FOO.
    Note that it is the whole output of FOO that passed to its child,
    not BAR.

    >>> s = 'FOO.BAR/BAZ'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.attrs_for_constraints
    ('BAR',)
    >>> token.flow_constraints
    ('BAZ',)
    >>> token.attrs_to_return  # the whole output is passed
    ()

    To pass a specific attributer if the output of a node satisfies
    a condition.
    E.g., only if the output of FOO equals to BAZ,
    its child gets FOO.QUX

    >>> s = 'FOO/BAZ=QUX'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.attrs_to_return
    ('QUX',)

    Combine the syntax above together, some simple but useful `if` logic
    can be constructed.
    The following example shows the output of `FRUIT`, or simply `FRUIT`,
    is passing `FRUIT.GRAPE` to its child if `FRUIT.COLOR==purple`.

    >>> s = 'FRUIT.COLOR/purple=GRAPE'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.node_name
    'FRUIT'
    >>> token.attrs_for_constraints
    ('COLOR',)
    >>> token.flow_constraints
    ('purple',)
    >>> token.attrs_to_return
    ('GRAPE',)

    To test if a attribute matches any of given constraints,
    use `,` to join those constraints.
    The following example shows the output of `FRUIT`, or simply `FRUIT`,
    is passing `FRUIT.GRAPE` to its child
    if `FRUIT.COLOR==purple` or `FRUIT.COLOR==green`.

    >>> s = 'FRUIT.COLOR/purple,green=GRAPE'
    >>> token = ParentSpecifierToken.parse_one(s)
    >>> token.node_name
    'FRUIT'
    >>> token.attrs_for_constraints
    ('COLOR',)
    >>> token.flow_constraints
    ('purple', 'green')
    >>> token.attrs_to_return
    ('GRAPE',)

    To define a tuple of tokens, use "|" to join those tokens,
    and parse it using `parse_many_tokens`.

    >>> s = 'FOO.BAR/BAZ=QUX | FUX/BOO'
    >>> tokens = ParentSpecifierToken.parse_many_tokens(s)
    >>> tokens[0].node_name
    'FOO'
    >>> tokens[1].node_name
    'FUX'

    To pass in a list of strings with each string defines a tuple of tokens,
    use `parse_list`.

    >>> s_list = ['FOO/BAR | FUX', 'BAZ.QUX']
    >>> tokens = ParentSpecifierToken.parse_list(s_list)
    >>> [token.node_name for token in tokens]
    ['FOO', 'FUX', 'BAZ']

    .. todo::
        Though currently a list of tokens is equivalent to
        a string of tokens join by "|".
        it has an oppurtunity to represent a logical-and,
        rather than a logical-or, i.e., "|".
        So that `['A|B', 'C', 'D|E|F']` represents
        "(A or B) and C and (D or E or F)".
    """

    node_name: str
    attrs_for_constraints: Tuple[str]
    flow_constraints: Tuple[Any]
    attrs_to_return: Tuple[str]

    @classmethod
    def parse_one(cls, s):
        """Parse a string representing a single token.

        E.g., A1.BC.DEF_123/ABC,DEFG=BCD.EDFG

        means:

            Under node A1, if BC.DEF_123 is ABC or DEFG then use BCD.EDFG
        """
        regex = (
            # node_name
            rf"\w+"
            # flow condition
            rf"({_Operator.REGEX_SUB}\w+)*"
            # flows
            rf"({_Operator.FLOW_IF}\w+({_Operator.FLOW_OR}\w+)*)?"
            # flow use
            rf"({_Operator.FLOW_USE}\w+({_Operator.REGEX_SUB}\w+)*)?")
        if not re.fullmatch(regex, s):
            raise SyntaxError(
                f"Each ParentSpecifierToken should match regex={regex}, got {s}"
            )
        s, _, attrs_to_return = s.partition(_Operator.FLOW_USE)
        attrs_to_return = (attrs_to_return.split(_Operator.SUB)
                           if attrs_to_return else [])
        s, _, flow_constraints = s.partition(_Operator.FLOW_IF)
        flow_constraints = (flow_constraints.split(_Operator.FLOW_OR)
                            if flow_constraints else [])
        node_name, *attrs_for_constraints = s.split(_Operator.SUB)

        # Originally, A.B.C is not using anything.
        # For convenience, we make
        # A.B.C alias to A.B.C=B.C
        # But for the followings, we let them be as they are.
        # So,
        # A.B.C/True is not alias to that
        # and A.B.C=D is not, either
        if len(attrs_to_return) == 0 and len(flow_constraints) == 0:
            attrs_to_return = attrs_for_constraints

        return cls(
            node_name,
            tuple(attrs_for_constraints),
            tuple(flow_constraints),
            tuple(attrs_to_return),
        )

    @classmethod
    def parse_many_tokens(cls, tokens: str) -> Tuple[ParentSpecifierToken]:
        """Parse a string representing several tokens joined by "|".

        E.g., A.B/123=C | D.E

        means:

            Under node A, if B==123 then use C or under node D, use E.
        """
        tokens = tokens.split(_Operator.OR)
        return tuple(cls.parse_one(s.strip()) for s in tokens)

    @classmethod
    def parse_list(cls, lists: Union[str,
                                     List[str]]) -> Tuple[ParentSpecifierToken]:
        """Parse a list of strings, each string representing several tokens.

        E.g. ['A.B/123=C | D.E', 'F']

        means:

            Under node A, if B==123 then use C, under node D, use E or use F.
        """
        if isinstance(lists, str):
            lists = [lists]
        return tuple(
            it.chain.from_iterable(cls.parse_many_tokens(s) for s in lists))


ParentSpecifier = Tuple[ParentSpecifierToken]
"""Typehint for a tuple of ParentSpecifierToken"""


class Parents:
    """Defines which nodes are parents.

    Also defines if it is to pass something to its child,
    what is being passed, and its child is using it as what.

    A conditional flow is also supported, by `ParentSpecifierToken`,
    i.e., to flow to it child only if some constraints are satisfied.
    """

    def __init__(self, args=None, kwargs=None):
        args: List[ParentSpecifier] = ([] if args is None else [
            ParentSpecifierToken.parse_list(name) for name in args
        ])
        kwargs: Dict[str, ParentSpecifier] = ({} if kwargs is None else {
            k: ParentSpecifierToken.parse_list(name)
            for k, name in kwargs.items()
        })
        self.node_map: Dict[str, List[Tuple[Union[
            int, str], ParentSpecifierToken]]] = defaultdict(list)
        self.parent_keys = set()
        for k, parent_specs in it.chain(enumerate(args), kwargs.items()):
            if k in self.parent_keys:
                raise SyntaxError(
                    "Parent key represents where the param passed from a"
                    " parent goes to, so it must be unique. Got multiple"
                    f" '{k}'")
            self.parent_keys.add(k)
            for spec_token in parent_specs:
                self.node_map[spec_token.node_name].append((k, spec_token))


class Senders(Parents):
    """Parents that pass something to its child.

    A general scheme is like:
    Receiver.FOO = Senders.BAR if Senders.BAZ == QUX

    The `Senders.BAR if Senders.BAZ == QUX` part is defined in
    `ParentSpecifierToken` and the `Receiver.FOO` part can either be
    defined positionally or by keyword.

    Use a node's output as the first input

    >>> s = 'FOO'
    >>> Senders.parse(s).node_map['FOO']
    [(0, ParentSpecifierToken(...))]

    Define senders positionally.

    >>> s = ['FOO', 'BAR']
    >>> senders = Senders.parse(s)
    >>> senders.node_map['FOO']
    [(0, ParentSpecifierToken(...))]
    >>> senders.node_map['BAR']
    [(1, ParentSpecifierToken(...))]

    Define senders by keywords.

    >>> s = {'a': 'FOO', 'b': 'BAR'}
    >>> senders = Senders.parse(s)
    >>> senders.node_map['FOO']
    [('a', ParentSpecifierToken(...))]
    >>> senders.node_map['BAR']
    [('b', ParentSpecifierToken(...))]

    To mix positional and keywords, use a list contains a dict.

    >>> s = ['FOO', 'BAR', {'a': 'BAZ', 'b': 'QUX'}]
    >>> senders = Senders.parse(s)
    >>> senders.node_map['FOO']
    [(0, ParentSpecifierToken(...))]
    >>> senders.node_map['BAR']
    [(1, ParentSpecifierToken(...))]
    >>> senders.node_map['BAZ']
    [('a', ParentSpecifierToken(...))]
    >>> senders.node_map['QUX']
    [('b', ParentSpecifierToken(...))]
    """

    @classmethod
    def parse(cls, o):
        if isinstance(o, dict):
            return cls([], o)
        if isinstance(o, list):
            args = []
            kwargs = {}
            for item in o:
                if isinstance(item, str):
                    args.append(item)
                elif isinstance(item, dict):
                    kwargs = {**item, **kwargs}
                else:
                    raise SyntaxError(
                        "Senders do not support parsed by list of list, got"
                        f" '{item}' in a list")
            return cls(args, kwargs)
        if isinstance(o, str):
            return cls([o], {})
        raise SyntaxError(
            f"Senders support parsed by only dict, list and str, got '{o}'")


class Callers(Parents):
    """Parents that triggers its child to start.

    A general scheme is like:

        Callee.start() if Callers.FOO == BAR

    THe `Callers.FOO == BAR` part is defined in
    `ParentSpecifierToken` and the `Callee.start` part is
    defined in Callee's tasks.

    >>> s = 'FOO'
    >>> Callers.parse(s).node_map['FOO']
    [(0, ParentSpecifierToken(...))]
    """

    @classmethod
    def parse(cls, o):
        if isinstance(o, str):
            return cls([o], {})
        if isinstance(o, list):
            return cls(o, {})
        raise SyntaxError(
            f"Callers support parsed by only str and list, got '{o}'")


class Node(BaseNode):
    """A node, with some tasks attached, of a directed, double-linked graph.

    `tasks attached` means a node has some specific tasks to do.
    `directed` means the relationship between two nodes is like
    parent-and-child rather than friend-and-friend.
    `double-linked` means the child can get to its parent, and the parent
    can also get to its child.

    A node is consistently listening to its parents and has a longer
    life time than a `NodeWorker`. If its parents are ready,
    the node will create a `NodeWorker` to run the attached list of `NodeTask`.

    A parent and its child communicate through a multiprocessing Queue.
    A node can receives some parameters from its parens, the senders.
    A node can be triggered by its parents, the callers,
    or as a starting point, for which the parent is defined as
    a special token, SWITCHON.
    The node will check whether the constraints are satisfied to decide
    to be triggered or not.
    If a node is triggered, it starts immediately without waiting for all of
    its senders passing their parameters to it.

    There are one or more tasks attached to a node, and the node is running
    them sequentially. The first task takes the parameters passed from
    the senders, and the second task takes the output of the first task,
    and so on.

    By the time that all tasks are done, the node sends the output of the
    final task to its receivers and callees.

    A node can be parsed using a node name, a valid structure held in a dict,
    a result queue and an end queue.

    >>> from multiprocessing import SimpleQueue
    >>> class MyTask(NodeTask):
    ...     def __init__(self, msg):
    ...         self.msg = msg
    ...     def main(self):
    ...         return self.msg
    >>> reset_node_tasks()
    >>> register_node_tasks(tasks=[MyTask])
    >>> end_q = SimpleQueue()
    >>> result_q = SimpleQueue()
    >>> simple_node = Node.parse(
    ...     name='HELLO', end_q=end_q, result_q=result_q, o=
    ...     {
    ...         'CALLERS': 'SWITCHON',
    ...         'TASK':
    ...         {
    ...             'MY_TASK': 'World'
    ...         },
    ...         'EXPORT': ['PRINT', 'RETURN'],
    ...         'END': True,
    ...     }
    ... )
    >>> simple_node.triggered_by(TriggerToken(), _S.SWITCHON)
    >>> simple_node.run()
    >>> # >>> HELLO: world
    >>> result_q.get()
    ('HELLO', 'World')

    Attributes:
        name: the name of a node.
            It should be unique within a flow to be used as a identifier.
        end: defines whether the node is the end point of the flow
            If so, the node send a termination message to all the nodes
            in the flow to stop the flow.
        tasks: a list of `NodeTask` to be done.
        end_q: a node-shared message queue to decide when to stop the flow.
            The end point of the flow sends message to this queue to notifies
            all nodes to stop.
        result_q: a node-shared message queue for flow to gather the results.
            If a node is defined to export its results to the flow returns,
            it sends its return value to this queue for the flow to gather
            the results.
        callers: a list of Callers.
        senders: a list of Senders.
        export: to export using print or to the flow returns.
    """

    def __init__(
        self,
        name: str,
        end: bool,
        tasks: List[NodeTask],
        end_q: SimpleQueue,
        result_q: SimpleQueue,
        callers: Union[None, Callers] = None,
        senders: Union[None, Senders] = None,
        export: Union[None, str, List[str]] = None,
    ):
        super().__init__()
        self.name = name
        self.tasks = tasks
        self.end = end
        self.callers = Callers() if callers is None else callers
        self.senders = Senders() if senders is None else senders
        self.receivers = []
        self.callees = []
        self.end_q = end_q
        self.result_q = result_q
        self.export = tuple() if export is None else export

        self._caller_qs: Dict[str, Queue] = {
            k: Queue() for k in self.callers.parent_keys
        }
        self._sender_qs: Dict[str, Queue] = {
            k: Queue() for k in self.senders.parent_keys
        }

    @staticmethod
    def _getitem(_p, _attr):
        """Use a lower case to access an attribute of a parameter.
        Lower cases is used because I think a case-insensitive parsing
        is more preferable. Furthermore, it's more pythonic to have
        a lower-case attributes in a class, so it should not affect
        many things.
        """
        _attr = _attr.lower()
        if isinstance(_p, dict):
            return _p[_attr]
        return _p.__getattribute__(_attr)

    def receive(self, param, sender_name):
        """Called by its sender to notify this node with the sender's result."""
        for parent_key, parent_spec in self.senders.node_map[sender_name]:

            # get attributes for constraints recursively
            # this is the same to
            # param."$attr[0]"."$attr[1]"...
            c = param
            for attr in parent_spec.attrs_for_constraints:
                c = self._getitem(c, attr)
            # get attributes to return recursively
            v = param
            for attr in parent_spec.attrs_to_return:
                v = self._getitem(v, attr)

            if len(parent_spec.flow_constraints) == 0:
                # if there's no constraints,
                # put the value into the sender's queue
                self._sender_qs[parent_key].put(v)

            # otherwise, put it only if there's any constraints satisfied.
            for flow in parent_spec.flow_constraints:
                if str(c) == flow:
                    self._sender_qs[parent_key].put(v)
                    break  # only need to put one

    def triggered_by(self, param, caller_name):
        """Called by its caller to notify this node with the caller's result.
        The caller's result is used for checking the constraints.
        """

        # triggered by unexpected caller is
        # currently considered fine, e.g. SWITCHON
        if caller_name not in self.callers.node_map:
            return
        for parent_key, parent_spec in self.callers.node_map[caller_name]:

            # get attributes for constraints recursively
            # this is the same to
            # param."$attr[0]"."$attr[1]"...
            c = param
            for attr in parent_spec.attrs_for_constraints:
                c = self._getitem(c, attr)

            # if there's no constraints,
            # put the TriggerToken into the sender's queue
            if len(parent_spec.flow_constraints) == 0:
                self._caller_qs[parent_key].put(TriggerToken())

            # otherwise, put it only if there's any constraints satisfied.
            for flow in parent_spec.flow_constraints:
                if str(c) == flow:
                    self._caller_qs[parent_key].put(TriggerToken())
                    break  # only need to put one

    def run(self) -> None:
        try:
            self.main()
        except Exception as err:
            logger.exception(err)
            self.end_q.put(EndOfFlowToken())
            raise err

    def main(self) -> None:
        meta = NodeMeta(
            self.name,
            self.callees,
            self.receivers,
            self.tasks,
            self.export,
            self.result_q,
            self.end_q,
        )
        worker_builder = NodeWorkerBuilder()
        worker_builder.meta = meta

        def check_queues(_q_vals, qs: Dict[str, Queue]):
            """Used to check if the given queue is ready"""
            _ok = True
            for key, q in qs.items():
                try:
                    # Use the latest value
                    _q_vals[key] = q.get(block=False)
                except Empty:
                    if key not in _q_vals:
                        _ok = False
            return _ok, _q_vals

        inputs = {}
        called = {}
        workers = []
        while self.end_q.empty():
            callers_ok, called = check_queues(called, self._caller_qs)
            senders_ok, inputs = check_queues(inputs, self._sender_qs)

            # if there's no callers defined, use senders as callers
            if len(self._caller_qs) == 0:
                callers_ok = senders_ok

            # if not getting called, sleep and try again
            if not callers_ok:
                time.sleep(0.1)
                continue

            # if callers ok but senders not ok, use whatever it have
            # args are the values with ordered integer keys in inputs
            args = [
                inputs.pop(i)
                for i in sorted(i for i in self.senders.parent_keys
                                if isinstance(i, int) and i in inputs)
            ]
            # kwargs are the values with string keys in inputs
            kwargs = {
                k: inputs.pop(k)
                for k in (k for k in self.senders.parent_keys
                          if isinstance(k, str) and k in inputs)
            }

            # reset called and inputs
            called = {}
            warn_left_keys(inputs)  # warn if there's left keys in inputs
            inputs = {}

            # create a worker to do the configured task
            # collect workers in a list to kill it if needed
            workers.append(worker_builder.build(*args, **kwargs))
            workers[-1].start()

            if self.end:
                for worker in workers:
                    worker.join()
                self.end_q.put(EndOfFlowToken())

            workers = [worker for worker in workers if worker.is_alive()]

        # Kill all workers alive.
        while True:
            for worker in workers:
                if worker.is_alive():
                    worker.kill()
                    logger.debug("killing worker {}", worker.name)
                    break
            else:
                break

    @classmethod
    def parse(cls, name, o, end_q: SimpleQueue, result_q: SimpleQueue):
        if name == _S.SWITCHON:
            raise SyntaxError(
                "SWITCHON is a reserved name, got a Node using it as its name")
        if not isinstance(o, dict):
            raise SyntaxError(f"Node support parsed by dict only, got {o}")
        tasks = None
        callers = None
        end = False
        senders = None
        export = None
        for k, v in o.items():
            if k == _S.TASK:
                tasks = NodeTask.parse(v)
            elif k == _S.CALLERS:
                callers = Callers.parse(v)
            elif k == _S.SENDERS:
                senders = Senders.parse(v)
            elif k == _S.END:
                end = bool(v)
            elif k == _S.EXPORT:
                if isinstance(v, str):
                    export = (v,)
                elif isinstance(v, list):
                    export = tuple(v)
                else:
                    raise SyntaxError(
                        f"export support parsed by only str and list, got {v}")
            else:
                raise SyntaxError(f"Unexpected Node member {k}")
        return cls(
            name=name,
            callers=callers,
            end=end,
            tasks=tasks,
            end_q=end_q,
            result_q=result_q,
            senders=senders,
            export=export,
        )


class Flow:
    """An entry point of a collection of Nodes.

    To create an instance of `Flow`, either `__init__`,
    `parse(dict)` or `parse_yaml(flow path or io stream)` can be used.

    >>> import io
    >>> import os
    >>> from multiprocessing import SimpleQueue
    >>> class MyTask(NodeTask):
    ...     def __init__(self, msg):
    ...         self.msg = msg
    ...     def main(self):
    ...         return self.msg
    >>> reset_node_tasks()
    >>> register_node_tasks(tasks=[MyTask])

    Using `__init__`.

    >>> end_q = SimpleQueue()
    >>> result_q = SimpleQueue()
    >>> simple_node = Node.parse(
    ...     name='HELLO', end_q=end_q, result_q=result_q, o=
    ...     {
    ...         'CALLERS': 'SWITCHON',
    ...         'TASK':
    ...         {
    ...             'MY_TASK': 'World'
    ...         },
    ...         'EXPORT': ['PRINT', 'RETURN'],
    ...         'END': True,
    ...     }
    ... )
    >>> flow_via_init = Flow([simple_node], result_q, end_q)

    Using `parse`.

    >>> obj = {
    ...     'HELLO':
    ...     {
    ...         'CALLERS': 'SWITCHON',
    ...         'TASK':
    ...         {
    ...             'MY_TASK': 'World'
    ...         },
    ...         'EXPORT': ['PRINT', 'RETURN'],
    ...         'END': True,
    ...     }
    ... }
    >>> flow_via_parse = Flow.parse(obj)

    Using `parse_yaml`.

    >>> yaml_str = os.linesep.join((
    ...     "FLOW: ",
    ...     "  HELLO:",
    ...     "    CALLERS: SWITCHON",
    ...     "    TASK:",
    ...     "      MY_TASK: World",
    ...     "    EXPORT:",
    ...     "      - PRINT",
    ...     "      - RETURN",
    ...     "    END: True",
    ... ))
    >>> flow_via_yaml = Flow.parse_yaml(io.StringIO(yaml_str))

    The above three definitions are equivalent and should give same results.

    >>> flow_via_init.start()
    [('HELLO', 'World')]
    >>> flow_via_parse.start()
    [('HELLO', 'World')]
    >>> flow_via_yaml.start()
    [('HELLO', 'World')]

    """

    def __init__(self, nodes: List[Node], result_q: SimpleQueue,
                 end_q: SimpleQueue):
        self.nodes = {}
        self.result_q = result_q
        self.end_q = end_q
        self.processes: List[Node] = []
        for node in nodes:
            if node.name in self.nodes:
                raise SyntaxError(f"Duplicated node name defined: {node.name}")
            self.nodes[node.name] = node

        for node in nodes:
            for name in node.callers.node_map.keys():
                if name == _S.SWITCHON:
                    continue
                self.nodes[name].callees.append(node)
            for name in node.senders.node_map.keys():
                self.nodes[name].receivers.append(node)

    @classmethod
    def parse(cls, o):
        if not isinstance(o, dict):
            raise SyntaxError(f"Flow only support parsed by dict, got {o}")
        end_q = SimpleQueue()
        result_q = SimpleQueue()
        nodes = [Node.parse(k, v, end_q, result_q) for k, v in o.items()]
        return cls(nodes, result_q, end_q)

    @classmethod
    def parse_yaml(cls, fp: Union[str, TextIO]):
        """Parse a yaml from either a path or an io-stream."""
        import yaml

        if isinstance(fp, str):
            with open_utf8(fp, "r") as f:
                main_dscp = yaml.safe_load(f)
        else:
            main_dscp = yaml.safe_load(fp)

        return cls.parse(main_dscp[_S.FLOW])

    def start(self):
        try:
            for node in self.nodes.values():
                self.processes.append(node)
                node.start()
            for node in self.nodes.values():
                node.triggered_by(TriggerToken(), _S.SWITCHON)
        except Exception as err:
            logger.exception(err)
            self.end_q.put(EndOfFlowToken())
            raise err

        for proc in self.processes:
            proc.join()
        results = []
        while not self.result_q.empty():
            results.append(self.result_q.get())
        return results


_NODE_TASK_MAPPING: Dict[str, Type[NodeTask]] = {}


def _camel_to_snake(s):
    """camel case to snake case.

    >>> _camel_to_snake('helloWorld')
    'hello_World'

    >>> _camel_to_snake('HelloWorld')
    'Hello_World'
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s)


def default_node_task_name(c: Type[NodeTask]):
    """Defines the default name of a node task.

    >>> class HelloWorld(NodeTask, abc.ABC):
    ...     pass
    >>> default_node_task_name(HelloWorld)
    'HELLO_WORLD'
    """
    return _camel_to_snake(c.__name__).upper()


def register_node_tasks(
    tasks: List[Type[NodeTask]] = None,
    task_map: Dict[str, Type[NodeTask]] = None,
    task_module: ModuleType = None,
    raise_on_exist: bool = True,
):
    """Used to register tasks.

    To register a task means to create a mapping from a name to a
    specific subclass derived from `NodeTask`, i.e. `sub-NodeTask`.

    Registering tasks is for `NodeTask.parse` to know which Task is used
    when a name is found while parsing.

    >>> class TaskFoo(NodeTask):
    ...     def main(self):
    ...         pass
    >>> class TaskBar(NodeTask):
    ...     def main(self):
    ...         pass

    There are three ways to register tasks,

    1. using a list of `sub-NodeTask`

        >>> reset_node_tasks()
        >>> register_node_tasks(tasks=[TaskFoo, TaskBar])
        >>> NodeTask.parse('TASK_FOO')[0].__class__.__name__
        'TaskFoo'

    2. using a dict from `str` to `sub-NodeTask`
        This way, a non-default node task name can be defined.

        >>> reset_node_tasks()
        >>> register_node_tasks(task_map={'MyFancy_name': TaskFoo})
        >>> NodeTask.parse('MyFancy_name')[0].__class__.__name__
        'TaskFoo'

    3. using an entire module of `sub-NodeTask`
        To register all `sub-NodeTask` in a module.
        Internally, every members of this module will be checked
        if it is a subclass of `NodeTask` and register it if true.

        .. highlight:: python
        .. code-block:: python

            import my_task_module
            register_node_tasks(my_task_module)

    If a name has be registered, ValueError is raised by default.

    >>> reset_node_tasks()
    >>> register_node_tasks(tasks=[TaskFoo])
    >>> register_node_tasks(task_map={'TASK_FOO': TaskBar})
    Traceback (most recent call last):
    ...
    ValueError: task name 'TASK_FOO' has already been registered

    To replace the old registered one without raising any error,
    pass in `raise_on_exist=False`.

    >>> reset_node_tasks()
    >>> register_node_tasks(tasks=[TaskFoo])
    >>> register_node_tasks(task_map={'TASK_FOO': TaskBar},
    ...                     raise_on_exist=False)
    >>> NodeTask.parse('TASK_FOO')[0].__class__.__name__
    'TaskBar'

    """
    tasks = [] if tasks is None else tasks
    task_map = {} if task_map is None else task_map

    def _check_before_update(_name, _task):
        if _name in _NODE_TASK_MAPPING and raise_on_exist:
            raise ValueError(f"task name '{_name}' has already been registered")
        _NODE_TASK_MAPPING[_name] = _task

    for task in tasks:
        name = default_node_task_name(task)
        _check_before_update(name, task)

    if task_module is not None:
        for task in task_module.__dict__.values():
            if (isinstance(task, type) and task is not NodeTask and
                    issubclass(task, NodeTask)):
                name = default_node_task_name(task)
                _check_before_update(name, task)

    for name, task in task_map.items():
        _check_before_update(name, task)


def reset_node_tasks():
    """Used to clear all registered node tasks.

    >>> reset_node_tasks()
    >>> _NODE_TASK_MAPPING['_foo'] = NodeTask
    >>> len(_NODE_TASK_MAPPING)
    1
    >>> reset_node_tasks()
    >>> len(_NODE_TASK_MAPPING)
    0

    """
    while len(_NODE_TASK_MAPPING) > 0:
        _NODE_TASK_MAPPING.popitem()
