import pymongo
from bson.objectid import ObjectId
from datatypes import TFlow
import os

MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017')
client = pymongo.MongoClient(MONGODB_URL)
db = client['loopflow-app']


class Dao:
    pass


class FlowFileDao(Dao):

    def save(self, file_name: str, file_content: str):
        file_id = db.flowfile.insert_one({
            "name": file_name,
            "content": file_content,
        }).inserted_id
        return str(file_id)

    def get(self, file_id):
        return db.flowfile.find_one({"_id": ObjectId(file_id)})


class FlowDao(Dao):

    def save(self, file_id: str, flow: TFlow):
        flow_id = db.flow.insert_one({
            "file_id": ObjectId(file_id),
            "flow": flow.dict()
        }).inserted_id
        return str(flow_id)

    def get(self, flow_id):
        return db.flow.find_one({"_id": ObjectId(flow_id)})
