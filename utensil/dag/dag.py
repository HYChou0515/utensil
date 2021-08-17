from __future__ import annotations

import datetime
from dataclasses import dataclass, field, InitVar
from typing import List, Dict, Type, Any, Union, Callable

from utensil.dag import dataflow
from utensil.general import warn_left_keys

try:
    import yaml
except ImportError as e:
    raise e

process_map: Dict[str, Type[dataflow.BaseNodeProcess]] = {
    'LOAD_DATA': dataflow.LoadData,
    'FILTER_ROWS': dataflow.FilterRows,
    'GET_FEATURE': dataflow.GetFeature,
    'LINEAR_NORMALIZE': dataflow.LinearNormalize,
    'MERGE_FEATURES': dataflow.MergeFeatures,
    'MAKE_DATASET': dataflow.MakeDataset,
    'MAKE_MODEL': dataflow.MakeModel,
    'GET_TARGET': dataflow.GetTarget,
    'CHANGE_TYPE_TO': dataflow.ChangeTypeTo,
    'TRAIN': dataflow.Train,
    'PREDICT': dataflow.Predict,
    'PARAMETER_SEARCH': dataflow.ParameterSearch,
    'SCORE': dataflow.Score,
}


class MISSING:
    pass


@dataclass
class DagNode:
    name: InitVar[str]
    parent_names: InitVar[List[str]]
    processes: InitVar[List[dataflow.BaseNodeProcess]]
    required: InitVar[bool]
    export: InitVar[Union[str, List[str]]]

    graph: Dag = None

    _name: str = None
    _processes: List[dataflow.BaseNodeProcess] = field(default_factory=list)
    _required: bool = None
    _export: List[Callable] = None

    _parent_nodes: Dict[str, dataflow.BaseNodeProcess] = field(default_factory=dict, init=False)
    _parent_results: Dict[str, Any] = field(default_factory=dict, init=False)
    _result: Any = field(default=MISSING, init=False)
    _is_dynamic: bool = field(default=MISSING, init=False)

    @property
    def name(self):
        return self._name

    @property
    def processes(self):
        return self._processes

    @property
    def parent_names(self):
        return self._parent_names

    @property
    def required(self):
        return self._required

    @property
    def result(self):
        return self._result

    @property
    def export(self):
        return self._export

    @property
    def is_dynamic(self):
        if self._is_dynamic is not MISSING:
            return self._is_dynamic
        for p in self.processes:
            if isinstance(p, dataflow.StatefulNodeProcess):
                self._is_dynamic = True
                return self._is_dynamic
        for pname in self.parent_names:
            if self.graph[pname].is_dynamic:
                self._is_dynamic = True
                return self._is_dynamic
        self._is_dynamic = False
        return False

    def __post_init__(self, name: str, parent_names: List[str], processes: List[dataflow.BaseNodeProcess],
                      required: bool, export: Union[str, List[str]]):
        self._name = name
        self._processes = processes
        self._parent_names = parent_names
        self._required = required
        self._export = []
        for exp in export if isinstance(export, list) else [export]:
            if exp == 'PRINT':
                self._export.append(print)
            else:
                raise ValueError

    def run(self):
        if self.result is MISSING or self.is_dynamic:
            node_inputs = [self.graph[parent].run() for parent in self.parent_names]
            start_time = datetime.datetime.now()
            if len(self._processes) > 0:
                # first process
                node_output = self._processes[0](*node_inputs)
                # second and after process
                for process in self._processes[1:]:
                    node_output = process(node_output)
            else:
                node_output = None
            self._result = node_output
            print(f'{self.name}: {datetime.datetime.now() - start_time}')
        else:
            print(f'{self.name}: cache read')
        for export in self.export:
            export(self.result)
        return self.result


@dataclass
class Dag:
    nodes: Dict[str, DagNode] = field(default_factory=dict)

    def __getitem__(self, item):
        return self.nodes[item]

    def _required_nodes(self):
        for node in self.nodes.values():
            if node.required:
                yield node

    def add_node(self, node):
        if node.name in self.nodes:
            raise ValueError
        if node.graph is not None:
            raise ValueError
        node.graph = self
        self.nodes[node.name] = node

    def run(self):
        for node in self._required_nodes():
            node.run()


dag_path = 'utensil/dag/covtype.dag'
dag_path = 'covtype.dag'
with open(dag_path, 'r') as f:
    dag_dscp = yaml.safe_load(f)

dag = Dag()
for node_name, node_dscp in dag_dscp['NODES'].items():
    parents = node_dscp.pop('PARENTS', [])  # empty parents means it is a starting point
    _processes = node_dscp.pop('PROCESSES', [])
    required = node_dscp.pop('REQUIRED', False)
    export = node_dscp.pop('EXPORT', [])

    warn_left_keys(node_dscp)

    node_processes = []
    for proc_dscp in _processes:
        if isinstance(proc_dscp, dict):
            if len(proc_dscp) != 1:
                raise ValueError
            pname, params = proc_dscp.popitem()
            process_class = process_map[pname]
            params = {} if params is None else params
        elif isinstance(proc_dscp, str):
            process_class = process_map[proc_dscp]
            params = {}
        process = process_class(params)  # noqa: process_class is a dataclasses
        node_processes.append(process)

    dag.add_node(DagNode(node_name, parent_names=parents, processes=node_processes,
                         required=required, export=export))

dag.run()
dag.run()
dag.run()
