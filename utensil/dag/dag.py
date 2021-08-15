from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import List, Dict, Type, Any

from utensil.dag import dataflow
from utensil.dag.helper import warn_left_keys

try:
    import yaml
except ImportError as e:
    raise e


process_map: Dict[str, Type[dataflow.NodeProcess]] = {
    'LOAD_DATA': dataflow.LoadData,
    'FILTER_ROWS': dataflow.FilterRows,
    'GET_FEATURE': dataflow.GetFeature,
    'LINEAR_NORMALIZE': dataflow.LinearNormalize,
    'MERGE_FEATURES': dataflow.MergeFeatures,
    'MAKE_DATASET': dataflow.MakeDataset,
    'GET_TARGET': dataflow.GetTarget,
    'TRAIN': dataflow.Train,
    'EVALUATE': dataflow.Evaluate,
}


@dataclass
class DagNode:
    name: InitVar[str]
    parent_names: InitVar[List[str]]
    processes: InitVar[List[dataflow.NodeProcess]]

    _name: str = None
    _parent_names: List[str] = field(default_factory=list)
    _processes: List[dataflow.NodeProcess] = field(default_factory=list)
    
    _children: List[DagNode] = field(default_factory=list, init=False)
    _parent_results: Dict[str, Any] = field(default_factory=dict, init=False)
        
    @property
    def name(self):
        return self._name

    @property
    def processes(self):
        return self._processes

    @property
    def children(self):
        return self._children

    @property
    def parent_names(self):
        return self._parent_names

    def __post_init__(self, name: str, parent_names: List[str], processes: List[dataflow.NodeProcess]):
        self._name = name
        self._processes = processes
        self._parent_names = parent_names

    def signal(self, name, result):
        # for a parent node inform this node that it is ready
        self._parent_results[name] = result
        if set(self._parent_results.keys()) == set(self.parent_names):
            self.run()

    def _build_input(self):
        if self.parent_names == ['ROOT']:
            return None
        node_input = []
        for parent_name in self.parent_names:
                node_input.append(self._parent_results[parent_name])
        return node_input

    def run(self):
        node_inputs = self._build_input()
        if len(self._processes) > 0:
            # first process
            if node_inputs is None:
                node_output = self._processes[0]()
            else:
                node_output = self._processes[0](*node_inputs)
            # second and after process
            for process in self._processes[1:]:
                node_output = process(node_output)
        else:
            node_output = None

        print(f'node={self.name} done')
        for child in self._children:
            child.signal(self.name, node_output)


@dataclass
class Dag:
    nodes: Dict[str, DagNode] = None

    def _starting_nodes(self):
        for node in self.nodes.values():
            if len(node.parent_names) == 0:
                yield node

    def run(self):
        for starting_node in self._starting_nodes():
            starting_node.run()


dag_path = 'utensil/dag/covtype.dag'
dag_path = 'covtype.dag'
with open(dag_path, 'r') as f:
    dag_dscp = yaml.safe_load(f)

nodes = {}
for node_name, node_dscp in dag_dscp['NODES'].items():
    parents = node_dscp.pop('PARENTS', [])  # empty parents means it is a starting point
    _processes = node_dscp.pop('PROCESSES', [])
    warn_left_keys(node_dscp)

    node_processes = []
    for proc_dscp in _processes:
        process_class = process_map[proc_dscp.pop('PROCESS')]
        process = process_class(proc_dscp.pop('PARAMETERS', {}))  # noqa: process_class is a dataclasses
        warn_left_keys(proc_dscp)
        node_processes.append(process)

    nodes[node_name] = DagNode(node_name, parent_names=parents, processes=node_processes)


for node in nodes.values():
    for parent_name in node.parent_names:
        nodes[parent_name].children.append(node)

dag = Dag(nodes)
dag.run()
