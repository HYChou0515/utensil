from abc import ABC, abstractmethod
from typing import Type

from bson import ObjectId
from pymongo.collection import Collection
from utensil import get_logger

from database import DBSessionMixin
from model import DbModel, MFlow, MFlowFile, MFlowJob

logger = get_logger(__file__)


class BasicCrud(ABC):

    @property
    @abstractmethod
    def default_collection(self) -> Collection:
        ...

    @property
    @abstractmethod
    def default_model(self) -> Type[DbModel]:
        ...

    async def create(self, model):
        new_model = await self.default_collection.insert_one(model.dict())
        model.id = new_model.inserted_id
        return model

    async def get(self, id):
        return self.default_model.parse_obj(
            await self.default_collection.find_one({"_id": ObjectId(id)}))


class FlowFileDao(DBSessionMixin, BasicCrud):

    @property
    def default_collection(self):
        return self.db.flowfile

    @property
    def default_model(self):
        return MFlowFile


class FlowDao(DBSessionMixin, BasicCrud):

    @property
    def default_collection(self):
        return self.db.flow

    @property
    def default_model(self):
        return MFlow


class FlowJobDao(DBSessionMixin, BasicCrud):

    @property
    def default_collection(self):
        return self.db.flowjob

    @property
    def default_model(self):
        return MFlowJob
