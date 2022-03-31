import itertools

from utensil.loopflow import loopflow
from utensil.loopflow.functions import basic, dataflow

from model.node_task import MNodeTaskListed


class Service:
    _all_modules = [basic, dataflow]

    def get_all_tasks_from_module(self, task_module):
        for task in task_module.__dict__.values():
            if (isinstance(task, type) and task is not loopflow.NodeTask and
                    issubclass(task, loopflow.NodeTask)):
                name = loopflow.default_node_task_name(task)
                yield name, task

    def get_all_tasks(self):
        return [
            MNodeTaskListed(key=name,
                            module=task.__module__,
                            task_name=task.__name__)
            for name, task in itertools.chain.from_iterable(
                self.get_all_tasks_from_module(module)
                for module in self._all_modules)
        ]
