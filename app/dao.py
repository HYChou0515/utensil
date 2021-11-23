from bson.objectid import ObjectId

from database import DBSessionMixin
from schema.datatypes import TFlow
from schema.flow_job import FlowJob, FlowJobInDb


class FlowFileDao(DBSessionMixin):

    def save(self, file_name: str, file_content: str):
        file_id = self.db.flowfile.insert_one({
            "name": file_name,
            "content": file_content,
        }).inserted_id
        return str(file_id)

    def get(self, file_id):
        return self.db.flowfile.find_one({"_id": ObjectId(file_id)})


class FlowDao(DBSessionMixin):

    def save(self, file_id: str, flow: TFlow):
        flow_id = self.db.flow.insert_one({
            "file_id": ObjectId(file_id),
            "flow": flow.dict()
        }).inserted_id
        return str(flow_id)

    def get(self, flow_id):
        return self.db.flow.find_one({"_id": ObjectId(flow_id)})


class FlowJobDao(DBSessionMixin):

    def create(self, flow_job: FlowJob):
        inserted_id = self.db.flowjob.insert_one(flow_job.dict()).inserted_id
        return FlowJobInDb(**flow_job.dict(), job_id=str(inserted_id))
