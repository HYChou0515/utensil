from dao import FlowDao, FlowFileDao
from datatypes import TFlow, TNode, TNodeTask

from utensil.loopflow.loopflow import (Flow, register_node_tasks,
                                       reset_node_tasks)


def load_flow(file_id):
    ...


def save_flow(file):
    file_name = file.filename
    file_content = file.file.read()
    flow_file_dao = FlowFileDao()
    file_id = flow_file_dao.save(file_name, file_content)
    return file_id


def parse_flow(file_id):
    flow_file_dao = FlowFileDao()
    flow_dao = FlowDao()
    d = flow_file_dao.get(file_id)
    reset_node_tasks()
    from utensil.loopflow.functions import basic, dataflow

    register_node_tasks(task_module=basic)
    register_node_tasks(task_module=dataflow)
    flow = Flow.parse_yaml(d["content"])
    tnodes = []
    for name, node in flow.nodes.items():
        tnodes.append(
            TNode(
                name=name,
                receivers=[receiver.name for receiver in node.receivers],
                callees=[
                    ":self:" if receiver.name == name else receiver.name
                    for receiver in node.callees
                ],
                end_of_flow=node.end,
                switchon="SWITCHON" in node.callers.node_map,
                tasks=[
                    TNodeTask(name=task.__class__.__name__)
                    for task in node.tasks
                ],
            ))
    tflow = TFlow(nodes=tnodes)
    flow_id = flow_dao.save(file_id, tflow)
    return flow_id, tflow
