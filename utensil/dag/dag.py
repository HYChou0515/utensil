from __future__ import annotations

import itertools
import re
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Process, Queue, SimpleQueue
from queue import Empty
from typing import Any, Dict, List, Optional, Tuple, Union


class BaseNode(Process):
    pass


class BaseNodeProcess(Process):
    pass


def camel_to_snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s)


class NodeProcessFunction:
    @classmethod
    def parse(cls, o):
        proc_map = {}
        for subc in cls.__subclasses__():
            proc_map[camel_to_snake(subc.__name__).upper()] = subc
        if isinstance(o, str):
            return proc_map[o]()
        elif isinstance(o, dict):
            if len(o) != 1:
                raise RuntimeError("E3")
            name, params = o.popitem()
            params = {k.lower(): v for k, v in params.items()}
            return proc_map[name](**params)  # noqa
        else:
            raise RuntimeError("E4")

    @abstractmethod
    def main(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        params = [kwargs.pop(k) for k in self.main.__code__.co_varnames if k in kwargs]
        return self.main(*params, *args, **kwargs)


@dataclass
class NodeProcessMeta:
    node_name: str
    triggerings: List[Node]
    children: List[Node]
    process_func: NodeProcessFunction


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
        super(self.__class__, self).__init__()
        self.meta: NodeProcessMeta = meta
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        print(self.meta.node_name, self.args, self.kwargs)
        ret = self.meta.process_func(*self.args, **self.kwargs)
        print(f"{self.meta.node_name}: {ret}")

        for triggering in self.meta.triggerings:
            triggering.trigger(ret, self.meta.node_name)
        for child in self.meta.children:
            child.push(ret, self.meta.node_name)


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
            rf"(\w+{_Operator.SUB})*\w+"
            rf"({_Operator.FLOW}\w+({_Operator.FLOW_OR}\w+)*)?"
            rf"({_Operator.FLOW_USE}\w+(.\w+)*)?",
            s,
        ):
            raise RuntimeError("E18")
        s, *flow_use = s.split(_Operator.FLOW_USE)
        flow_condition, *flows = s.split(_Operator.FLOW)
        flows = [] if len(flows) == 0 else flows[0].split(_Operator.FLOW_OR)
        node_name, *flow_condition = flow_condition.split(_Operator.SUB)
        return cls(node_name, tuple(flow_condition), tuple(flows), tuple(flow_use))


class ParentSpecifiers(tuple):
    @classmethod
    def parse(cls, s):
        return cls(
            ParentSpecifier.parse(spec.strip()) for spec in s.split(_Operator.OR)
        )


class Parents:
    def __init__(self, args=None, kwargs=None):
        self.args: List[ParentSpecifiers] = (
            [] if args is None else [ParentSpecifiers.parse(name) for name in args]
        )
        self.kwargs: Dict[str, ParentSpecifiers] = (
            {}
            if kwargs is None
            else {k: ParentSpecifiers.parse(name) for k, name in kwargs.items()}
        )
        self.node_map = defaultdict(list)
        self.parent_keys = set()
        for k, parent_specs in itertools.chain(
            enumerate(self.args), self.kwargs.items()
        ):
            if k in self.parent_keys:
                raise RuntimeError("E19")
            self.parent_keys.add(k)
            for spec in parent_specs:
                self.node_map[spec.node_name].append((k, spec))

    @classmethod
    def parse(cls, o):
        if isinstance(o, dict):
            return cls([], o)
        elif isinstance(o, list):
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
        else:
            raise RuntimeError("E2")


class Triggers(Parents):
    @classmethod
    def parse(cls, o):
        if isinstance(o, str):
            return cls([o], {})
        elif isinstance(o, list):
            return cls(o, {})
        else:
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
        proc_func: NodeProcessFunction,
        end_q: SimpleQueue,
        triggers: Union[None, Triggers] = None,
        parents: Union[None, Parents] = None,
    ):
        super(self.__class__, self).__init__()
        self.name = name
        self.proc_func = proc_func
        self.end = end
        self.triggers = Triggers() if triggers is None else triggers
        self.parents = Parents() if parents is None else parents
        self.children = []
        self.triggerings = []
        self.end_q = end_q

        self._tqs: Dict[str, Queue] = {k: Queue() for k in self.triggers.parent_keys}
        self._qs: Dict[str, Queue] = {k: Queue() for k in self.parents.parent_keys}

    @staticmethod
    def _getitem(_p, _attr):
        _attr = _attr.lower()
        if isinstance(_p, dict):
            return _p[_attr]
        else:
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
        # triggered by unexpected caller is currently considered fine, e.g. SWITCHON
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
            self.name, self.triggerings, self.children, self.proc_func
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
            args = [inputs.pop(i) for i in range(len(self.parents.args)) if i in inputs]
            kwargs = {
                k: inputs.pop(k) for k in self.parents.kwargs.keys() if k in inputs
            }
            proc = process_builder.build(*args, **kwargs)
            proc.start()

            if self.end:
                self.end_q.put(object)

    @classmethod
    def parse(cls, name, o, end_q: SimpleQueue):
        if name == SWITCHON:
            raise RuntimeError("E21")
        if not isinstance(o, dict):
            raise RuntimeError("E5")
        proc_func = None
        triggers = None
        end = False
        parents = None
        for k, v in o.items():
            if k == "PROCESS":
                proc_func = NodeProcessFunction.parse(v)
            elif k == "TRIGGERS":
                triggers = Triggers.parse(v)
            elif k == "PARENTS":
                parents = Parents.parse(v)
            elif k == "END":
                if v:
                    end = True
            else:
                raise RuntimeError("E13")
        return cls(
            name=name,
            triggers=triggers,
            end=end,
            proc_func=proc_func,
            end_q=end_q,
            parents=parents,
        )


class Dag:
    def __init__(self, nodes: List[Node]):
        self.nodes = {}
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
        nodes = [Node.parse(k, v, end_q) for k, v in o.items()]
        return cls(nodes)

    @classmethod
    def parse_yaml(cls, dag_path):
        try:
            import yaml
        except ImportError as e:
            raise e

        with open(dag_path, "r") as f:
            main_dscp = yaml.safe_load(f)

        return cls.parse(main_dscp["DAG"])

    def start(self):
        for node in self.nodes.values():
            self.processes.append(node)
            node.start()
        for node in self.nodes.values():
            node.trigger(TriggerToken(), SWITCHON)
        for proc in self.processes:
            proc.join()
