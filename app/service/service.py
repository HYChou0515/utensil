import inspect
import itertools
import json
from copy import deepcopy
from importlib import import_module
from typing import List, Any

import yaml
from utensil.loopflow import loopflow
from utensil.loopflow.functions import basic, dataflow
from utensil.loopflow.loopflow import reset_node_tasks, register_node_tasks, Flow

from model.flow_graph import MFlowGraph, MFlowGraphLink, MFlowGraphNode
from model.node_task import MNodeTaskListed


class Service:
    _all_modules = [basic, dataflow]

    def __init__(self):
        super().__init__()
        self._module_map = {m.__name__: m for m in self._all_modules}

    def get_all_tasks_from_module(self, task_module):
        for task in task_module.__dict__.values():
            if (isinstance(task, type) and task is not loopflow.NodeTask and
                    issubclass(task, loopflow.NodeTask)):
                name = loopflow.default_node_task_name(task)
                yield name, task

    def get_all_tasks(self) -> List[MNodeTaskListed]:
        """List all node tasks from all default modules.

        >>> service = Service()
        >>> service.get_all_tasks()
        [MNodeTaskListed(...), ...]

        :return: list of node tasks
        """
        ret = []

        for name, task in itertools.chain.from_iterable(
                self.get_all_tasks_from_module(module)
                for module in self._all_modules):
            arg_names = list(inspect.getfullargspec(task.main).args[1:])
            if (varargs :=
                    inspect.getfullargspec(task.main).varargs) is not None:
                arg_names.append(f"*{varargs}")
            signature = inspect.signature(task.__init__)
            params = []
            for k, v in signature.parameters.items():
                if k == 'self':
                    continue
                if v.kind in (v.VAR_POSITIONAL, v.KEYWORD_ONLY, v.VAR_KEYWORD):
                    params.append((k, 'optional'))
                elif v.default is inspect.Parameter.empty:
                    params.append((k, 'required'))
                else:
                    params.append((k, 'optional'))
            ret.append(
                MNodeTaskListed(key=name,
                                module=task.__module__,
                                task_name=task.__name__,
                                arg_names=arg_names,
                                params=params))
        return ret

    def get_source_code_of_node_task(self, module: str, task_name: str) -> str:
        """Get source code of a given task in a module.

        >>> service = Service()
        >>> service.get_source_code_of_node_task("utensil.loopflow.functions.basic", "Dummy")
        'class Dummy(NodeTask):...'

        >>> service.get_source_code_of_node_task("foo", "bar")
        Traceback (most recent call last):
        ...
        KeyError: 'module foo, task bar not found'

        :param module: the name of the module
        :param task_name: the name of the task
        :return: the source code of the given task in the module
        """
        try:
            m = self._module_map[module]
            for _, task in self.get_all_tasks_from_module(m):
                if task.__name__ == task_name:
                    return inspect.getsource(task)
        except KeyError:
            pass
        raise KeyError(f'module {module}, task {task_name} not found')

    def create_graph(self, graph_body: MFlowGraph):
        """Create a loop flow from a graph.

        :param graph_body: a serialized graph
        :return:
        """

        def id_to_name(_id):
            return f"ID_{_id.replace('-', '_')}"

        reset_node_tasks()
        flow: dict[str, Any] = {}
        links = {}
        task_map = {}

        switch_on_id = None
        eof_id = None
        for layer in graph_body.layers:
            for model in layer.models.values():
                if isinstance(model, MFlowGraphLink):
                    links[model.id] = model
                if isinstance(model, MFlowGraphNode):
                    if model.node_type == 'switch-on':
                        switch_on_id = model.id
                    if model.node_type == 'end-of-flow':
                        eof_id = model.id
        for layer in graph_body.layers:
            for model in layer.models.values():
                if isinstance(model,
                              MFlowGraphNode) and model.node_type == 'task':
                    model_name = id_to_name(model.id)
                    task_cls = import_module(model.module)
                    task_map[f'{model.module}.{model.name}'] = getattr(
                        task_cls, model.name)
                    params = {
                        par_name: json.loads(var)
                        for var, (par_name,
                                  _) in zip(model.param_values, model.params)
                    }
                    senders = {}
                    callers_str = None
                    eof = False
                    for port in model.ports:
                        if port.is_in:
                            if port.name == 'trigger':
                                callers = []
                                for lnk_id in port.links:
                                    if links[lnk_id].source == switch_on_id:
                                        callers.append('SWITCHON')
                                    else:
                                        callers.append(
                                            id_to_name(links[lnk_id].source))
                                if len(callers) > 0:
                                    callers_str = '|'.join(callers)
                            elif len(port.links) > 0:
                                senders[port.name] = '|'.join([
                                    id_to_name(links[lnk_id].source)
                                    for lnk_id in port.links
                                ])
                        else:
                            if port.name == 'out':
                                for lnk_id in port.links:
                                    if links[lnk_id].target == eof_id:
                                        eof = True

                    flow[model_name] = {
                        'EXPORT': ['PRINT', 'RETURN'],
                    }
                    if len(senders) > 0:
                        flow[model_name]['SENDERS'] = senders
                    if len(params) > 0:
                        flow[model_name]['TASK'] = {
                            f'{model.module}.{model.name}': params
                        }
                    else:
                        flow[model_name][
                            'TASK'] = f'{model.module}.{model.name}'
                    if callers_str is not None:
                        flow[model_name]['CALLERS'] = callers_str
                    if eof:
                        flow[model_name]['END'] = True

        register_node_tasks(task_map=task_map)
        Flow.parse(deepcopy(flow))  # try flow is valid
        return yaml.dump(flow)
