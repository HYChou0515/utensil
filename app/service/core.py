from datetime import datetime

from utensil.loopflow.loopflow import (Flow, register_node_tasks,
                                       reset_node_tasks)

from dao import FlowDao, FlowFileDao, FlowJobDao
from database import DBSessionMixin
from schema.datatypes import TFlow, TNode, TNodeTask
from schema.flow_job import FlowJob, FlowJobCreate


class FlowService(DBSessionMixin):

    def load_flow(self, file_id):
        ...

    def save_flow(self, file):
        file_name = file.filename
        file_content = file.file.read()
        flow_file_dao = FlowFileDao(self.db)
        file_id = flow_file_dao.save(file_name, file_content)
        return file_id

    def parse_flow(self, file_id):
        flow_file_dao = FlowFileDao(self.db)
        flow_dao = FlowDao(self.db)
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


class FlowJobService(DBSessionMixin):

    def create_flow_job(self, flow_job_create: FlowJobCreate):
        flow_job_dao = FlowJobDao(self.db)
        flow_job = FlowJob(**flow_job_create.dict(),
                           created_time=datetime.now())
        flow_job_in_db = flow_job_dao.create(flow_job)
        return flow_job_in_db
