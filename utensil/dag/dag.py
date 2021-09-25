from __future__ import annotations

import datetime
import itertools
import re
import time
from abc import abstractmethod
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field, InitVar
from enum import Enum
from multiprocessing import Process, Queue, SimpleQueue
from queue import Empty
from threading import Thread
from typing import List, Dict, Type, Any, Union, Callable, Tuple, Optional

from utensil.dag import dataflow
from utensil.general import warn_left_keys

try:
    import yaml
except ImportError as e:
    raise e


class BaseNode(Process):
    pass


class BaseNodeProcess(Process):
    pass


def camel_to_snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s)


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
                raise RuntimeError('E3')
            name, params = o.popitem()
            params = {k.lower(): v for k, v in params.items()}
            return proc_map[name](**params)
        else:
            raise RuntimeError('E4')

    @abstractmethod
    def main(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        params = [kwargs.pop(k) for k in self.main.__code__.co_varnames if k in kwargs]
        return self.main(*params, *args, **kwargs)


@dataclass
class NodeProcessMeta:
    node_name: str
    children: List[Node]
    process_func: NodeProcessFunction


class NodeProcessBuilder:
    def __init__(self):
        self.meta: Optional[NodeProcessMeta] = None

    def build(self, *args, **kwargs):
        if self.meta is None:
            raise RuntimeError('E12')
        kwargs = {k.lower(): v for k, v in kwargs.items()}
        proc = NodeProcess(self.meta, *args, **kwargs)
        return proc


class NodeProcess(BaseNodeProcess):
    def __init__(self, meta, *args, **kwargs):
        super(self.__class__, self).__init__()
        self.meta = meta
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        print(self.meta.node_name, self.args, self.kwargs)
        ret = self.meta.process_func(*self.args, **self.kwargs)
        print(f'{self.meta.node_name}: {ret}')
        for child in self.meta.children:
            child.push(ret, self.meta.node_name)


# region NodeProcessFunction Impl


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


# endregion

DEFAULT_FLOW = object()


class _Operator(str, Enum):
    OR = '|'
    FLOW = '/'
    FLOW_OR = ','
    SUB = '.'
    FLOW_USE = '='


@dataclass(frozen=True)
class ParentSpecifier:
    node_name: str
    flow_condition: Tuple[str]
    flows: Tuple[Any]
    flow_use: Tuple[str]

    @classmethod
    def parse(cls, s):
        # s should be like A1.BC.DEF_123/ABC,DEFG=BCD.EDFG
        if not re.fullmatch(rf'(\w+{_Operator.SUB})*\w+({_Operator.FLOW}\w+({_Operator.FLOW_OR}\w+)*)?({_Operator.FLOW_USE}\w+(.\w+)*)?', s):
            raise RuntimeError('E18')
        s, *flow_use = s.split(_Operator.FLOW_USE)
        flow_condition, *flows = s.split(_Operator.FLOW)
        flows = [] if len(flows) == 0 else flows[0].split(_Operator.FLOW_OR)
        node_name, *flow_condition = flow_condition.split(_Operator.SUB)
        return cls(node_name, tuple(flow_condition), tuple(flows), tuple(flow_use))


class ParentSpecifiers(tuple):
    @classmethod
    def parse(cls, s):
        return cls(ParentSpecifier.parse(spec.strip()) for spec in s.split(_Operator.OR))


class Parents:
    def __init__(self, args=None, kwargs=None):
        self.args: List[ParentSpecifiers] = [] if args is None else [ParentSpecifiers.parse(name) for name in args]
        self.kwargs: Dict[str, ParentSpecifiers] = {} if kwargs is None else {k: ParentSpecifiers.parse(name) for k, name in kwargs.items()}
        self.node_map = defaultdict(list)
        self.parent_keys = set()
        for k, parent_specs in itertools.chain(enumerate(self.args), self.kwargs.items()):
            if k in self.parent_keys:
                raise RuntimeError('E19')
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
                    raise RuntimeError('E1')
            print(args, kwargs)
            return cls(args, kwargs)
        else:
            raise RuntimeError('E2')


class Node(BaseNode):
    def __init__(self, name: str, init: bool, end: bool,
                 proc_func: NodeProcessFunction,
                 end_q: SimpleQueue,
                 parents: Union[None, Parents] = None):
        super(self.__class__, self).__init__()
        self.name = name
        self.proc_func = proc_func
        self.init = init
        self.end = end
        self.parents = Parents() if parents is None else parents
        self.children = []
        self.end_q = end_q

        self._qs: Dict[str, Queue] = {k: Queue() for k in self.parents.parent_keys}

    def push(self, param, caller_name):
        def getitem(_p, _attr):
            _attr = _attr.lower()
            if isinstance(_p, dict):
                return _p[_attr]
            else:
                return _p.__getattribute__(_attr)

        for parent_key, parent_spec in self.parents.node_map[caller_name]:
            c = param
            for attr in parent_spec.flow_condition:
                c = getitem(c, attr)
            v = param
            for attr in parent_spec.flow_use:
                v = getitem(v, attr)

            if len(parent_spec.flows) == 0:
                self._qs[parent_key].put(v)
            for flow in parent_spec.flows:
                if str(c) == flow:
                    self._qs[parent_key].put(v)
                    break  # only need to put one

    def run(self) -> None:
        meta = NodeProcessMeta(self.name, self.children, self.proc_func)
        process_builder = NodeProcessBuilder()
        process_builder.meta = meta

        if self.init:
            proc = process_builder.build()
            proc.start()
        if len(self._qs) == 0:
            return

        def build_inputs(_inputs):
            _ok = True
            for parent_key, q in self._qs.items():
                if parent_key in _inputs:
                    # got this param already
                    continue
                try:
                    _inputs[parent_key] = q.get(block=False)
                except Empty:
                    _ok = False
            return _ok, _inputs

        inputs = {}
        while self.end_q.empty():
            ok, inputs = build_inputs(inputs)
            if not ok:
                time.sleep(0.1)
                continue
            args = [inputs.pop(i) for i in range(len(self.parents.args))]
            kwargs = {k: inputs.pop(k) for k in self.parents.kwargs.keys()}
            proc = process_builder.build(*args, **kwargs)
            proc.start()

            if self.end:
                self.end_q.put(object)

    @classmethod
    def parse(cls, name, o, end_q: SimpleQueue):
        if not isinstance(o, dict):
            raise RuntimeError('E5')
        proc_func = None
        init = False
        end = False
        parents = None
        for k, v in o.items():
            if k == 'PROCESS':
                proc_func = NodeProcessFunction.parse(v)
            elif k == 'INIT':
                if v:
                    init = True
            elif k == 'PARENTS':
                parents = Parents.parse(v)
            elif k == 'END':
                if v:
                    end = True
            else:
                raise RuntimeError('E13')
        return cls(name=name, init=init, end=end, proc_func=proc_func, end_q=end_q, parents=parents)


class Dag:
    def __init__(self, nodes: List[Node]):
        self.nodes = {}
        self.init_nodes = []
        self.processes: List[Process] = []
        for node in nodes:
            if node.name in self.nodes:
                raise RuntimeError('E6')
            self.nodes[node.name] = node
            if node.init:
                self.init_nodes.append(node.name)

        for node in nodes:
            for name in node.parents.node_map.keys():
                self.nodes[name].children.append(node)

    @classmethod
    def parse(cls, o):
        if not isinstance(o, dict):
            raise RuntimeError('E7')
        end_q = SimpleQueue()
        nodes = [Node.parse(k, v, end_q) for k, v in o.items()]
        return cls(nodes)

    def start(self):
        for node in self.nodes.values():
            self.processes.append(node)
            node.start()
        for proc in self.processes:
            proc.join()


dag_path = 'utensil/dag/simple.dag'
dag_path = 'simple.dag'
with open(dag_path, 'r') as f:
    main_dscp = yaml.safe_load(f)

dag = Dag.parse(main_dscp['DAG'])
dag.start()
node_info = {}
