import io
from datetime import datetime

from fastapi import Depends
from utensil.general.logger import get_logger
from utensil.loopflow.loopflow import (Flow, register_node_tasks,
                                       reset_node_tasks)

from dao import FlowDao, FlowFileDao, FlowJobDao
from model import (JobStatusState, MFlow, MFlowFile, MFlowJob,
                   MFlowJobCreateByFlow, MJobStatus, MNode, MNodeStatus,
                   MNodeTask, NodeStatusState)

logger = get_logger(__file__)


class FlowService:

    def __init__(
            self,
            flow_dao: FlowDao = Depends(FlowDao),
            flow_file_dao: FlowFileDao = Depends(FlowFileDao),
    ):
        self.flow_dao = flow_dao
        self.flow_file_dao = flow_file_dao

    async def load_flow(self, flow_id) -> MFlow:
        return MFlow.parse_obj(await self.flow_dao.get(flow_id))

    async def save_flow(self, file) -> MFlowFile:
        file_name = file.filename
        file_content = file.file.read()
        flow_file = MFlowFile(name=file_name, content=file_content)
        flow_file_in_db = await self.flow_file_dao.create(flow_file)
        return flow_file_in_db

    async def parse_flow(self, file_id) -> MFlow:
        reset_node_tasks()
        from utensil.loopflow.functions import basic, dataflow

        register_node_tasks(task_module=basic)
        register_node_tasks(task_module=dataflow)
        flow_file_in_db = await self.flow_file_dao.get(file_id)
        flow = Flow.parse_yaml(io.StringIO(flow_file_in_db.content))
        tnodes = []
        for name, node in flow.nodes.items():
            tnodes.append(
                MNode(
                    name=name,
                    receivers=[receiver.name for receiver in node.receivers],
                    callees=[
                        ":self:" if receiver.name == name else receiver.name
                        for receiver in node.callees
                    ],
                    end_of_flow=node.end,
                    switchon="SWITCHON" in node.callers.node_map,
                    tasks=[
                        MNodeTask(name=task.__class__.__name__)
                        for task in node.tasks
                    ],
                ))
        flow_model = MFlow(file_id=file_id, nodes=tnodes)
        flow_in_db = await self.flow_dao.create(flow_model)
        return flow_in_db


class FlowJobService:

    def __init__(
            self,
            flow_service: FlowService = Depends(FlowService),
            flow_job_dao: FlowJobDao = Depends(FlowJobDao),
    ):
        self.flow_service = flow_service
        self.flow_job_dao = flow_job_dao

    async def create_flow_job(
            self, flow_job_create: MFlowJobCreateByFlow) -> MFlowJob:
        flow = await self.flow_service.load_flow(flow_job_create.flow_id)
        node_status = {
            node.name: MNodeStatus(current_task=node.tasks[0].name,
                                   state=NodeStatusState.WAIT)
            for node in flow.nodes
        }
        flow_job = MFlowJob(**flow_job_create.dict(),
                            created_time=datetime.now(),
                            status=MJobStatus(state=JobStatusState.INITIALIZED),
                            node_status=node_status)
        flow_job_in_db = await self.flow_job_dao.create(flow_job)
        return flow_job_in_db
