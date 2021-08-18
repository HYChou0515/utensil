from __future__ import annotations

import datetime
from collections import namedtuple
from dataclasses import dataclass, field, InitVar
from typing import List, Dict, Type, Any, Union, Callable, Tuple

from utensil.dag import dataflow
from utensil.general import warn_left_keys

try:
    import yaml
except ImportError as e:
    raise e

SWITCHON = 'SWITCHON'

process_map: Dict[str, Type[dataflow.BaseNodeProcess]] = {
    'LOAD_DATA': dataflow.LoadData,
    'FILTER_ROWS': dataflow.FilterRows,
    'GET_FEATURE': dataflow.GetFeature,
    'GET_ITEM': dataflow.GetItem,
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
    'SAMPLING_ROWS': dataflow.SamplingRows,
    'STATE_UPDATE': dataflow.StateUpdate,
}


class MISSING:
    pass


@dataclass
class DemandPort:
    name: str
    demand_node: ProcessNode = MISSING
    _demand_dag_name: str = field(init=False, repr=False)
    _demand_node_name: str = field(init=False, repr=False)

    @classmethod
    def from_dscp(cls, name: str, dscp: Dict):
        if len(dscp) != 1:
            raise SyntaxError(f'should be exactly one demand node for a demand port')
        demand_port = cls(name)
        demand_port._demand_dag_name, demand_port._demand_node_name = dscp.popitem()
        return demand_port

    def compile_demand_node(self, dag_map: Dict[str, Dag]):
        self.demand_node = dag_map[self._demand_dag_name].supplies[self._demand_node_name].node
        del self._demand_dag_name
        del self._demand_node_name


@dataclass
class SupplyPort:
    node: Union[str, ProcessNode]


@dataclass
class ProcessNode:
    name: InitVar[str]
    parents: InitVar[List[Union[str, ProcessNode, DemandPort]]]
    processes: InitVar[List[dataflow.BaseNodeProcess]]
    triggered_by: InitVar[Union[str, List[DemandPort]]]
    export: InitVar[Union[str, List[str]]]

    _name: str = field(init=False)
    _processes: List[dataflow.BaseNodeProcess] = field(default_factory=list, init=False, repr=False)
    _triggered_by: Union[str, List[DemandPort]] = field(default_factory=list, init=False, repr=False)
    _export: List[Callable] = field(default_factory=list, init=False, repr=False)

    _parents: List[Union[str, ProcessNode]] = field(default_factory=list, init=False, repr=False)
    _parent_results: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _result: Any = field(default=MISSING, init=False, repr=False)
    _is_dynamic: bool = field(default=MISSING, init=False, repr=False)

    @property
    def name(self):
        return self._name

    @property
    def processes(self):
        return self._processes

    @property
    def parents(self):
        return self._parents

    @property
    def parent_names(self):
        return [p.name for p in self.parents]

    @property
    def triggered_by(self):
        return self._triggered_by

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
        if len(self.triggered_by) > 0:
            self._is_dynamic = True
            return self._is_dynamic
        for parent in self.parents:
            if parent.is_dynamic:
                self._is_dynamic = True
                return self._is_dynamic
        self._is_dynamic = False
        return False

    def __post_init__(self, name: str, parents: List[Union[str, ProcessNode]],
                      processes: List[dataflow.BaseNodeProcess],
                      triggered_by: Union[str, List[DemandPort]], export: Union[str, List[str]]):
        self._name = name
        self._parents = parents
        self._processes = processes
        self._triggered_by = triggered_by
        for exp in export if isinstance(export, list) else [export]:
            if exp == 'PRINT':
                self._export.append(print)
            else:
                raise ValueError

    def _print(self, s):
        pass

    def run(self):
        if self.result is MISSING or self.is_dynamic:
            node_inputs = [parent.run() for parent in self.parents]
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
            self._print(f'{self.name}: {datetime.datetime.now() - start_time}')
        else:
            self._print(f'{self.name}: cache read')
        for export in self.export:
            export(self.result)
        return self.result

    @classmethod
    def from_dscp(cls, name: str, dscp: Dict[str, Any]):
        parents = dscp.pop('PARENTS', [])  # empty parents means it is a starting point
        processes = dscp.pop('PROCESSES', [])
        triggered_by = dscp.pop('TRIGGERED_BY', [])
        export = dscp.pop('EXPORT', [])

        warn_left_keys(dscp)

        node_processes = []
        for proc_dscp in [processes] if isinstance(processes, str) else processes:
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
        return cls(name, parents=parents, processes=node_processes,
                   triggered_by=triggered_by, export=export)


@dataclass
class Dag:
    name: str
    supplies: Dict[str, Union[str, SupplyPort]]
    demands: Dict[str, Union[str, DemandPort]]
    nodes: Dict[str, Union[str, ProcessNode]]

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

    def compile_demand_port(self, dag_map: Dict[str, Dag]):
        for demand in self.demands.values():
            demand.compile_demand_node(dag_map)

    @classmethod
    def from_dscp(cls, name: str, dscp: Dict):
        supplies = {name: SupplyPort(name) for name in dscp.pop('SUPPLIES', [])}
        demands = {name: DemandPort.from_dscp(name, dscp) for name, dscp in dscp.pop('DEMANDS', {}).items()}
        nodes = {name: ProcessNode.from_dscp(name, dscp) for name, dscp in dscp.pop('NODES', {}).items()}
        warn_left_keys(dscp)

        # syntax check
        # node names and demand names should be disjoint
        if not nodes.keys().isdisjoint(demands.keys()):
            raise SyntaxError('node names and demand names are not disjoint')
        # SWITCHON should be reserved
        if SWITCHON in nodes or SWITCHON in demands:
            raise SyntaxError(f'\'{SWITCHON}\' is a reserved for triggers')

        # compile supplies
        for supply in supplies.values():
            if supply.node in nodes:
                supply.node = nodes[supply.node]
            else:
                raise KeyError(f'supply name \'{supply.node}\' not found in nodes')
        # compile nodes
        for node in nodes.values():
            for i, parent_name in enumerate(node.parents):
                if parent_name in nodes:
                    node.parents[i] = nodes[parent_name]
                elif parent_name in demands:
                    node.parents[i] = demands[parent_name]
                else:
                    raise KeyError(f'node parent \'{parent_name}\' not found in nodes and demands')
            for i, trigger_name in enumerate(node.triggered_by):
                if trigger_name in demands:
                    node.triggered_by[i] = demands[trigger_name]
                elif trigger_name == SWITCHON:
                    pass
                else:
                    raise KeyError(f'trigger \'{trigger_name}\' is not \'{SWITCHON}\' and not found in demands')

        return cls(name, supplies, demands, nodes)


dag_path = 'utensil/dag/covtype.dag'
dag_path = 'covtype.dag'
with open(dag_path, 'r') as f:
    main_dscp = yaml.safe_load(f)

dags: Dict[str, Dag] = {}
for dag_name, dag_dscp in main_dscp.pop('DAGS', {}).items():
    if dag_name in dags:
        raise SyntaxError(f'duplicate dag name {dag_name} in DAG definition')
    dags[dag_name] = Dag.from_dscp(dag_name, dag_dscp)
for dag in dags.values():
    dag.compile_demand_port(dags)
