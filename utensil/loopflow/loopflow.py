from __future__ import annotations

import itertools
from sys import platform
import re
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue, SimpleQueue, set_start_method
from queue import Empty
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from utensil import get_logger
from utensil.general import open_utf8

logger = get_logger(__name__)

if platform == 'darwin':
    set_start_method("fork")


class BaseNode(Process):
    pass


class BaseNodeProcess(Process):
    pass


class NodeProcessFunction:

    @classmethod
    def parse(cls, o) -> List[NodeProcessFunction]:
        proc_map = _NODE_PROCESS_MAPPING

        def _parse_1(_o):
            if isinstance(_o, str):
                return proc_map[_o]()
            if isinstance(_o, dict):
                if len(_o) != 1:
                    raise RuntimeError("E3")
                name, params = _o.popitem()
                logger.debug(name, params)
                if isinstance(params, list):
                    return proc_map[name](*params)  # noqa
                if isinstance(params, dict):
                    params = {k.lower(): v for k, v in params.items()}
                    return proc_map[name](**params)  # noqa
                print(proc_map[name])
                return proc_map[name](params)  # noqa
            raise RuntimeError("E4")

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
class NodeProcessMeta:
    node_name: str
    triggerings: List[Node]
    children: List[Node]
    process_funcs: List[NodeProcessFunction]
    export: Tuple[str]
    result_q: SimpleQueue
    end_q: SimpleQueue


class NodeProcessBuilder:

    def __init__(self):
        self.meta: Optional[NodeProcessMeta] = None

    def build(self, *args, **kwargs):
        if self.meta is None:
            raise RuntimeError("E12")
        kwargs = {k.lower(): v for k, v in kwargs.items()}
        proc = NodeProcess(self.meta, *args, **kwargs)
        return proc


class NodeProcess(BaseNodeProcess):

    def __init__(self, meta: NodeProcessMeta, *args, **kwargs):
        super().__init__()
        self.meta: NodeProcessMeta = meta
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        logger.debug('Node {} started', self.meta.node_name)
        try:
            ret = self.meta.process_funcs[0](*self.args, **self.kwargs)
            for proc_func in self.meta.process_funcs[1:]:
                ret = proc_func(ret)
            if "PRINT" in self.meta.export:
                print(f"{self.meta.node_name}: {ret}")
            if "RETURN" in self.meta.export:
                self.meta.result_q.put((self.meta.node_name, ret))
            for triggering in self.meta.triggerings:
                triggering.trigger(ret, self.meta.node_name)
            for child in self.meta.children:
                child.push(ret, self.meta.node_name)
        except Exception as err:
            logger.exception(err)
            self.meta.end_q.put(object())
            raise err


DEFAULT_FLOW = object()


class _Operator(str, Enum):
    OR = "|"
    FLOW = "/"
    FLOW_OR = ","
    SUB = "."
    FLOW_USE = "="


@dataclass(frozen=True)
class ParentSpecifier:
    node_name: str
    flow_condition: Tuple[str]
    flows: Tuple[Any]
    flow_use: Tuple[str]

    @classmethod
    def parse(cls, s):
        # s should be like A1.BC.DEF_123/ABC,DEFG=BCD.EDFG
        if not re.fullmatch(
                rf"\w+"  # node_name
                rf"({_Operator.SUB}\w+)*"  # flow condition
                rf"({_Operator.FLOW}\w+({_Operator.FLOW_OR}\w+)*)?"  # flows
                rf"({_Operator.FLOW_USE}\w+({_Operator.SUB}\w+)*)?",  # flow use
                s,
        ):
            raise RuntimeError("E18")
        s, _, flow_use = s.partition(_Operator.FLOW_USE)
        flow_use = flow_use.split(_Operator.SUB) if flow_use else []
        s, _, flows = s.partition(_Operator.FLOW)
        flows = flows.split(_Operator.FLOW_OR) if flows else []
        node_name, *flow_condition = s.split(_Operator.SUB)

        # A.B.C is alias to A.B.C=B.C
        # A.B.C/True is not
        # A.B.C=D is not
        if len(flow_use) == 0 and len(flows) == 0:
            flow_use = flow_condition

        return cls(node_name, tuple(flow_condition), tuple(flows),
                   tuple(flow_use))


class ParentSpecifiers(tuple):

    @classmethod
    def parse(cls, list_str: Union[str, List[str]]):
        if isinstance(list_str, str):
            list_str = [list_str]
        return cls(
            tuple(
                ParentSpecifier.parse(spec.strip())
                for spec in s.split(_Operator.OR))
            for s in list_str)


class Parents:

    def __init__(self, args=None, kwargs=None):
        self.args: List[ParentSpecifiers] = ([] if args is None else [
            ParentSpecifiers.parse(name) for name in args
        ])
        self.kwargs: Dict[str, ParentSpecifiers] = ({} if kwargs is None else {
            k: ParentSpecifiers.parse(name) for k, name in kwargs.items()
        })
        self.node_map = defaultdict(list)
        self.parent_keys = set()
        for k, parent_specs in itertools.chain(enumerate(self.args),
                                               self.kwargs.items()):
            if k in self.parent_keys:
                raise RuntimeError("E19")
            self.parent_keys.add(k)
            for specs in parent_specs:
                for spec in specs:
                    self.node_map[spec.node_name].append((k, spec))

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
                    raise RuntimeError("E1")
            return cls(args, kwargs)
        if isinstance(o, str):
            return cls([o], {})
        raise RuntimeError("E2")


class Triggers(Parents):

    @classmethod
    def parse(cls, o):
        if isinstance(o, str):
            return cls([o], {})
        if isinstance(o, list):
            return cls(o, {})
        raise RuntimeError("E20")


SWITCHON = "SWITCHON"


@dataclass
class TriggerToken:
    pass


class Node(BaseNode):

    def __init__(
        self,
        name: str,
        end: bool,
        proc_funcs: List[NodeProcessFunction],
        end_q: SimpleQueue,
        result_q: SimpleQueue,
        triggers: Union[None, Triggers] = None,
        parents: Union[None, Parents] = None,
        export: Union[None, str, List[str]] = None,
    ):
        super().__init__()
        self.name = name
        self.proc_funcs = proc_funcs
        self.end = end
        self.triggers = Triggers() if triggers is None else triggers
        self.parents = Parents() if parents is None else parents
        self.children = []
        self.triggerings = []
        self.end_q = end_q
        self.result_q = result_q
        if export is None:
            self.export = tuple()
        elif isinstance(export, str):
            self.export = (export,)
        elif isinstance(export, list):
            self.export = tuple(export)
        else:
            raise RuntimeError("E24")

        self._tqs: Dict[str, Queue] = {
            k: Queue() for k in self.triggers.parent_keys
        }
        self._qs: Dict[str,
                       Queue] = {k: Queue() for k in self.parents.parent_keys}

    @staticmethod
    def _getitem(_p, _attr):
        _attr = _attr.lower()
        if isinstance(_p, dict):
            return _p[_attr]
        return _p.__getattribute__(_attr)

    def push(self, param, caller_name):
        for parent_key, parent_spec in self.parents.node_map[caller_name]:
            c = param
            for attr in parent_spec.flow_condition:
                c = self._getitem(c, attr)
            v = param
            for attr in parent_spec.flow_use:
                v = self._getitem(v, attr)

            if len(parent_spec.flows) == 0:
                self._qs[parent_key].put(v)
            for flow in parent_spec.flows:
                if str(c) == flow:
                    self._qs[parent_key].put(v)
                    break  # only need to put one

    def trigger(self, param, caller_name):
        # triggered by unexpected caller is
        # currently considered fine, e.g. SWITCHON
        if caller_name not in self.triggers.node_map:
            return
        for parent_key, parent_spec in self.triggers.node_map[caller_name]:
            c = param
            for attr in parent_spec.flow_condition:
                c = self._getitem(c, attr)
            if len(parent_spec.flows) == 0:
                self._tqs[parent_key].put(TriggerToken())
            for flow in parent_spec.flows:
                if str(c) == flow:
                    self._tqs[parent_key].put(TriggerToken())
                    break  # only need to put one

    def run(self) -> None:
        meta = NodeProcessMeta(
            self.name,
            self.triggerings,
            self.children,
            self.proc_funcs,
            self.export,
            self.result_q,
            self.end_q,
        )
        process_builder = NodeProcessBuilder()
        process_builder.meta = meta

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
        triggered = {}
        procs = []
        while self.end_q.empty():
            triggers_ok, triggered = check_queues(triggered, self._tqs)
            parents_ok, inputs = check_queues(inputs, self._qs)

            # if there's no triggers defined, use parents as triggers
            if len(self._tqs) == 0:
                triggers_ok = parents_ok

            # if not getting triggered, sleep and try again
            if not triggers_ok:
                time.sleep(0.1)
                continue

            # if getting triggered, reset triggers but not inputs
            triggered = {}

            # if triggers ok but parents not ok, use whatever it have
            args = [
                inputs.pop(i)
                for i in range(len(self.parents.args))
                if i in inputs
            ]
            kwargs = {
                k: inputs.pop(k)
                for k in self.parents.kwargs.keys()
                if k in inputs
            }
            procs.append(process_builder.build(*args, **kwargs))
            procs[-1].start()

            if self.end:
                for proc in procs:
                    proc.join()
                self.end_q.put(object())

            procs = [proc for proc in procs if proc.is_alive()]

        while True:
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
                    logger.debug("killing proc {}", proc.name)
                    break
            else:
                break

    @classmethod
    def parse(cls, name, o, end_q: SimpleQueue, result_q: SimpleQueue):
        if name == SWITCHON:
            raise RuntimeError("E21")
        if not isinstance(o, dict):
            raise RuntimeError("E5")
        proc_funcs = None
        triggers = None
        end = False
        parents = None
        export = None
        for k, v in o.items():
            if k == "PROCESS":
                proc_funcs = NodeProcessFunction.parse(v)
            elif k == "TRIGGERS":
                triggers = Triggers.parse(v)
            elif k == "PARENTS":
                parents = Parents.parse(v)
            elif k == "END":
                end = bool(v)
            elif k == "EXPORT":
                export = v
            else:
                raise RuntimeError(f"E13: {k}")
        return cls(
            name=name,
            triggers=triggers,
            end=end,
            proc_funcs=proc_funcs,
            end_q=end_q,
            result_q=result_q,
            parents=parents,
            export=export,
        )


class Flow:

    def __init__(self, nodes: List[Node], result_q: SimpleQueue):
        self.nodes = {}
        self.result_q = result_q
        self.processes: List[Process] = []
        for node in nodes:
            if node.name in self.nodes:
                raise RuntimeError("E6")
            self.nodes[node.name] = node

        for node in nodes:
            for name in node.triggers.node_map.keys():
                if name == SWITCHON:
                    continue
                self.nodes[name].triggerings.append(node)
            for name in node.parents.node_map.keys():
                self.nodes[name].children.append(node)

    @classmethod
    def parse(cls, o):
        if not isinstance(o, dict):
            raise RuntimeError("E7")
        end_q = SimpleQueue()
        result_q = SimpleQueue()
        nodes = [Node.parse(k, v, end_q, result_q) for k, v in o.items()]
        return cls(nodes, result_q)

    @classmethod
    def parse_yaml(cls, flow_path):
        import yaml

        with open_utf8(flow_path, "r") as f:
            main_dscp = yaml.safe_load(f)

        return cls.parse(main_dscp["FLOW"])

    def start(self):
        for node in self.nodes.values():
            self.processes.append(node)
            node.start()
        for node in self.nodes.values():
            node.trigger(TriggerToken(), SWITCHON)
        for proc in self.processes:
            proc.join()
        results = []
        while not self.result_q.empty():
            results.append(self.result_q.get())
        return results


_NODE_PROCESS_MAPPING: Dict[str, Type[NodeProcessFunction]] = {}


def camel_to_snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s)


def default_node_process_function_name(c: Type[NodeProcessFunction]):
    return camel_to_snake(c.__name__).upper()


def reset_node_process_functions():
    while len(_NODE_PROCESS_MAPPING) > 0:
        _NODE_PROCESS_MAPPING.popitem()


def register_node_process_functions(
    proc_funcs: List[Type[NodeProcessFunction]] = None,
    proc_func_map: Dict[str, Type[NodeProcessFunction]] = None,
    proc_func_module: ModuleType = None,
    raise_on_exist: bool = True,
):
    proc_funcs = [] if proc_funcs is None else proc_funcs
    proc_func_map = {} if proc_func_map is None else proc_func_map

    def _check_before_update(_name, _proc_func):
        if _name in _NODE_PROCESS_MAPPING and raise_on_exist:
            raise RuntimeError(f"E25 {_name}")
        _NODE_PROCESS_MAPPING[_name] = _proc_func

    for proc_func in proc_funcs:
        name = default_node_process_function_name(proc_func)
        _check_before_update(name, proc_func)

    if proc_func_module is not None:
        for proc_func in proc_func_module.__dict__.values():
            if (isinstance(proc_func, type) and
                    proc_func is not NodeProcessFunction and
                    issubclass(proc_func, NodeProcessFunction)):
                name = default_node_process_function_name(proc_func)
                _check_before_update(name, proc_func)

    for name, proc_func in proc_func_map.items():
        _check_before_update(name, proc_func)
