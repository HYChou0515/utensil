from __future__ import annotations

import itertools as it
import re
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue, SimpleQueue, set_start_method
from queue import Empty
from sys import platform
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from utensil import get_logger
from utensil.general import open_utf8, warn_left_keys

logger = get_logger(__name__)

if platform == "darwin":
    set_start_method("fork")


class _SyntaxToken(str, Enum):
    # Flow tokens
    FLOW = "FLOW"
    # Node tokens
    SENDERS = "PARENTS"
    CALLERS = "TRIGGERS"
    TASK = "PROCESS"
    EXPORT = "EXPORT"
    END = "END"
    # Special tokens
    SWITCHON = "SWITCHON"


_S = _SyntaxToken


class _ExportOperator(str, Enum):
    PRINT = "PRINT"
    RETURN = "RETURN"


class BaseNode(Process):
    """A base class for Node"""


class BaseNodeWorker(Process):
    """A base class for NodeWorker"""


class NodeTask:

    @classmethod
    def parse(cls, o) -> List[NodeTask]:
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
                print(task_map[name])
                return task_map[name](params)  # noqa
            raise SyntaxError(
                f"Task item support parsed by only str and dict, got {o}")

        if not isinstance(o, list):
            o = [o]

        return [_parse_1(_) for _ in o]

    @abstractmethod
    def main(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        params = [
            kwargs.pop(k) for k in self.main.__code__.co_varnames if k in kwargs
        ]
        return self.main(*params, *args, **kwargs)


@dataclass
class NodeMeta:
    node_name: str
    callees: List[Node]
    receivers: List[Node]
    tasks: List[NodeTask]
    export: Tuple[str]
    result_q: SimpleQueue
    end_q: SimpleQueue


class NodeWorkerBuilder:

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
            for callee in self.meta.callees:
                callee.triggered_by(ret, self.meta.node_name)
            for receiver in self.meta.receivers:
                receiver.receive(ret, self.meta.node_name)
        except Exception as err:
            logger.exception(err)
            self.meta.end_q.put(object())
            raise err


class _Operator(str, Enum):
    OR = "|"
    FLOW_IF = "/"
    FLOW_OR = ","
    SUB = "."
    FLOW_USE = "="


@dataclass(frozen=True)
class ParentSpecifierToken:
    node_name: str
    node_property: Tuple[str]
    flow_if: Tuple[Any]
    flow_use: Tuple[str]

    @classmethod
    def parse_one(cls, s):
        # s should be like A1.BC.DEF_123/ABC,DEFG=BCD.EDFG
        # means:
        # Under A1, if BC.DEF_123 is ABC or DEFG then use BCD.EDFG
        regex = (
            rf"\w+"  # node_name
            rf"({_Operator.SUB}\w+)*"  # flow condition
            rf"({_Operator.FLOW_IF}\w+({_Operator.FLOW_OR}\w+)*)?"  # flows
            rf"({_Operator.FLOW_USE}\w+({_Operator.SUB}\w+)*)?"  # flow use
        )
        if not re.fullmatch(regex, s):
            raise SyntaxError(
                f"Each ParentSpecifierToken should match regex={regex}, got {s}"
            )
        s, _, flow_use = s.partition(_Operator.FLOW_USE)
        flow_use = flow_use.split(_Operator.SUB) if flow_use else []
        s, _, flow_if = s.partition(_Operator.FLOW_IF)
        flow_if = flow_if.split(_Operator.FLOW_OR) if flow_if else []
        node_name, *node_property = s.split(_Operator.SUB)

        # Originally, A.B.C is not using anything.
        # For convenience, we make
        # A.B.C alias to A.B.C=B.C
        # But for the followings, we let them be as they are.
        # So,
        # A.B.C/True is not alias to that
        # and A.B.C=D is not, either
        if len(flow_use) == 0 and len(flow_if) == 0:
            flow_use = node_property

        return cls(node_name, tuple(node_property), tuple(flow_if),
                   tuple(flow_use))

    @classmethod
    def parse_many_token(cls, tokens: str) -> Tuple[ParentSpecifierToken]:
        tokens = tokens.split(_Operator.OR)
        return tuple(cls.parse_one(s.strip()) for s in tokens)

    @classmethod
    def parse_many_list(
            cls, lists: Union[str, List[str]]) -> Tuple[ParentSpecifierToken]:
        if isinstance(lists, str):
            lists = [lists]
        return tuple(
            it.chain.from_iterable(cls.parse_many_token(s) for s in lists))


ParentSpecifier = Tuple[ParentSpecifierToken]


class Parents:

    def __init__(self, args=None, kwargs=None):
        args: List[ParentSpecifier] = ([] if args is None else [
            ParentSpecifierToken.parse_many_list(name) for name in args
        ])
        kwargs: Dict[str, ParentSpecifier] = ({} if kwargs is None else {
            k: ParentSpecifierToken.parse_many_list(name)
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

    @classmethod
    def parse(cls, o):
        if isinstance(o, str):
            return cls([o], {})
        if isinstance(o, list):
            return cls(o, {})
        raise SyntaxError(
            f"Callers support parsed by only str and list, got '{o}'")


class TriggerToken:
    pass


class Node(BaseNode):

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
        _attr = _attr.lower()
        if isinstance(_p, dict):
            return _p[_attr]
        return _p.__getattribute__(_attr)

    def receive(self, param, sender_name):
        for parent_key, parent_spec in self.senders.node_map[sender_name]:
            c = param
            for attr in parent_spec.node_property:
                c = self._getitem(c, attr)
            v = param
            for attr in parent_spec.flow_use:
                v = self._getitem(v, attr)

            if len(parent_spec.flow_if) == 0:
                self._sender_qs[parent_key].put(v)
            for flow in parent_spec.flow_if:
                if str(c) == flow:
                    self._sender_qs[parent_key].put(v)
                    break  # only need to put one

    def triggered_by(self, param, caller_name):
        # param is used for checking condition

        # triggered by unexpected caller is
        # currently considered fine, e.g. SWITCHON
        if caller_name not in self.callers.node_map:
            return
        for parent_key, parent_spec in self.callers.node_map[caller_name]:
            c = param
            for attr in parent_spec.node_property:
                c = self._getitem(c, attr)
            if len(parent_spec.flow_if) == 0:
                self._caller_qs[parent_key].put(TriggerToken())
            for flow in parent_spec.flow_if:
                if str(c) == flow:
                    self._caller_qs[parent_key].put(TriggerToken())
                    break  # only need to put one

    def run(self) -> None:
        try:
            self.main()
        except Exception as err:
            logger.exception(err)
            self.end_q.put(object())
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
            _ok = True
            for key, q in qs.items():
                if key in _q_vals:
                    # got this key already
                    continue
                try:
                    _q_vals[key] = q.get(block=False)
                except Empty:
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

            # if getting called, reset called but not inputs
            called = {}

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
            warn_left_keys(inputs)

            workers.append(worker_builder.build(*args, **kwargs))
            workers[-1].start()

            if self.end:
                for worker in workers:
                    worker.join()
                self.end_q.put(object())

            workers = [worker for worker in workers if worker.is_alive()]

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
    def parse_yaml(cls, flow_path):
        import yaml

        with open_utf8(flow_path, "r") as f:
            main_dscp = yaml.safe_load(f)

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
            self.end_q.put(object())
            raise err

        for proc in self.processes:
            proc.join()
        results = []
        while not self.result_q.empty():
            results.append(self.result_q.get())
        return results


_NODE_TASK_MAPPING: Dict[str, Type[NodeTask]] = {}


def camel_to_snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s)


def default_node_task_name(c: Type[NodeTask]):
    return camel_to_snake(c.__name__).upper()


def reset_node_tasks():
    while len(_NODE_TASK_MAPPING) > 0:
        _NODE_TASK_MAPPING.popitem()


def register_node_tasks(
    tasks: List[Type[NodeTask]] = None,
    task_map: Dict[str, Type[NodeTask]] = None,
    task_module: ModuleType = None,
    raise_on_exist: bool = True,
):
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
