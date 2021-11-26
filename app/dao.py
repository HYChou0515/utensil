from bson.objectid import ObjectId
from utensil import get_logger

from database import DBSessionMixin
from model import MFlow, MFlowFile, MFlowJob

logger = get_logger(__file__)


class FlowFileDao(DBSessionMixin):

    async def create(self, flow_file: MFlowFile) -> MFlowFile:
        new_obj = await self.db.flowfile.insert_one(flow_file.dict())
        flow_file.id = new_obj.inserted_id
        return flow_file

    async def get(self, file_id) -> MFlowFile:
        return MFlowFile.parse_obj(await self.db.flowfile.find_one(
            {"_id": ObjectId(file_id)}))


class FlowDao(DBSessionMixin):

    async def create(self, flow: MFlow) -> MFlow:
        new_obj = await self.db.flow.insert_one(flow.dict())
        flow.id = new_obj.inserted_id
        return flow

    async def get(self, flow_id) -> MFlow:
        return MFlow.parse_obj(await
                               self.db.flow.find_one({"_id": ObjectId(flow_id)}
                                                    ))


class FlowJobDao(DBSessionMixin):

    async def create(self, flow_job: MFlowJob) -> MFlowJob:
        new_obj = await self.db.flowjob.insert_one(flow_job.dict())
        flow_job.id = new_obj.inserted_id
        return flow_job
