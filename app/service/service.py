import inspect
import itertools
from typing import List

from utensil.loopflow import loopflow
from utensil.loopflow.functions import basic, dataflow

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
