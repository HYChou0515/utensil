from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Union

from bson import ObjectId
from pydantic import BaseModel, Field


class Model(BaseModel):
    pass


class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class DbModel(Model):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class MFlowFile(DbModel):
    name: str
    content: str


class MNodeTask(Model):
    name: str


class MNode(Model):
    name: str
    receivers: list[Union[MNode, str]]
    callees: list[Union[MNode, str]]
    end_of_flow: bool
    switchon: bool
    tasks: list[MNodeTask]


MNode.update_forward_refs()


class MFlow(DbModel):
    file_id: PyObjectId
    nodes: list[MNode]

    class Config(DbModel.Config):
        schema_extra = {
            "example": {
                "_id":
                    "61a1959f9e55743c5404174f",
                "file_id":
                    "61a1959f9e55743c5404174d",
                "nodes": [
                    {
                        "name":
                            "COUNTER",
                        "receivers": ["COUNTER"],
                        "callees": ["TERMINAL", "MODEL_PARAM", "RAW_DATA"],
                        "end_of_flow":
                            False,
                        "switchon":
                            True,
                        "tasks": [
                            {
                                "name": "Default"
                            },
                            {
                                "name": "Add"
                            },
                            {
                                "name": "LessEqual"
                            },
                        ],
                    },
                    {
                        "name": "TERMINAL",
                        "receivers": [],
                        "callees": [],
                        "end_of_flow": True,
                        "switchon": False,
                        "tasks": [{
                            "name": "Dummy"
                        }],
                    },
                    {
                        "name": "MODEL_PARAM",
                        "receivers": ["RAW_MODEL"],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "RandomParameterSearch"
                        }],
                    },
                    {
                        "name": "RAW_MODEL",
                        "receivers": ["TRAINED_MODEL"],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "MakeModel"
                        }],
                    },
                    {
                        "name": "RAW_DATA",
                        "receivers": [
                            "FEATURE_ELEVATION", "FEATURE_ASPECT", "TARGET"
                        ],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": True,
                        "tasks": [{
                            "name": "LoadData"
                        }, {
                            "name": "FilterRows"
                        }],
                    },
                    {
                        "name":
                            "FEATURE_ELEVATION",
                        "receivers": ["FEATURES"],
                        "callees": [],
                        "end_of_flow":
                            False,
                        "switchon":
                            False,
                        "tasks": [{
                            "name": "GetFeature"
                        }, {
                            "name": "LinearNormalize"
                        }],
                    },
                    {
                        "name":
                            "FEATURE_ASPECT",
                        "receivers": ["FEATURES"],
                        "callees": [],
                        "end_of_flow":
                            False,
                        "switchon":
                            False,
                        "tasks": [{
                            "name": "GetFeature"
                        }, {
                            "name": "LinearNormalize"
                        }],
                    },
                    {
                        "name": "FEATURES",
                        "receivers": ["DATASET"],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "MergeFeatures"
                        }],
                    },
                    {
                        "name":
                            "TARGET",
                        "receivers": ["DATASET"],
                        "callees": [],
                        "end_of_flow":
                            False,
                        "switchon":
                            False,
                        "tasks": [{
                            "name": "GetTarget"
                        }, {
                            "name": "ChangeTypeTo"
                        }],
                    },
                    {
                        "name":
                            "DATASET",
                        "receivers": ["TRAIN_DATA", "TEST_DATA"],
                        "callees": [],
                        "end_of_flow":
                            False,
                        "switchon":
                            False,
                        "tasks": [{
                            "name": "MakeDataset"
                        }, {
                            "name": "SamplingRows"
                        }],
                    },
                    {
                        "name": "TRAIN_DATA",
                        "receivers": ["TRAINED_MODEL", "TRAIN_SCORE"],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "Dummy"
                        }],
                    },
                    {
                        "name": "TEST_DATA",
                        "receivers": ["TEST_SCORE"],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "Dummy"
                        }],
                    },
                    {
                        "name": "TRAINED_MODEL",
                        "receivers": ["TRAIN_SCORE", "TEST_SCORE"],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "Train"
                        }],
                    },
                    {
                        "name": "TRAIN_SCORE",
                        "receivers": [],
                        "callees": [],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "Score"
                        }],
                    },
                    {
                        "name": "TEST_SCORE",
                        "receivers": [],
                        "callees": ["COUNTER"],
                        "end_of_flow": False,
                        "switchon": False,
                        "tasks": [{
                            "name": "Score"
                        }],
                    },
                ],
            }
        }


class JobStatusState(str, Enum):
    INITIALIZED = "initialized"
    PROCESSING = "processing"
    FINISHED = "finished"
    ERROR = "error"


class MJobStatus(Model):
    state: JobStatusState


class NodeStatusState(str, Enum):
    RUN = "run"
    WAIT = "wait"


class MNodeStatus(Model):
    current_task: str
    state: NodeStatusState


class MFlowJobCreateByFlow(Model):
    flow_id: str


class MFlowJob(DbModel):
    flow_id: PyObjectId
    created_time: datetime
    status: MJobStatus
    node_status: dict[str, MNodeStatus]

    class Config(DbModel.Config):
        schema_extra = {
            "example": {
                "_id": "61a195fb9e55743c54041751",
                "flow_id": "61a1959f9e55743c5404174f",
                "created_time": "2021-11-27T10:20:43.581606",
                "status": {
                    "state": "initialized"
                },
                "node_status": {
                    "COUNTER": {
                        "current_task": "Default",
                        "state": "wait"
                    },
                    "TERMINAL": {
                        "current_task": "Dummy",
                        "state": "wait"
                    },
                    "MODEL_PARAM": {
                        "current_task": "RandomParameterSearch",
                        "state": "wait",
                    },
                    "RAW_MODEL": {
                        "current_task": "MakeModel",
                        "state": "wait"
                    },
                    "RAW_DATA": {
                        "current_task": "LoadData",
                        "state": "wait"
                    },
                    "FEATURE_ELEVATION": {
                        "current_task": "GetFeature",
                        "state": "wait",
                    },
                    "FEATURE_ASPECT": {
                        "current_task": "GetFeature",
                        "state": "wait"
                    },
                    "FEATURES": {
                        "current_task": "MergeFeatures",
                        "state": "wait"
                    },
                    "TARGET": {
                        "current_task": "GetTarget",
                        "state": "wait"
                    },
                    "DATASET": {
                        "current_task": "MakeDataset",
                        "state": "wait"
                    },
                    "TRAIN_DATA": {
                        "current_task": "Dummy",
                        "state": "wait"
                    },
                    "TEST_DATA": {
                        "current_task": "Dummy",
                        "state": "wait"
                    },
                    "TRAINED_MODEL": {
                        "current_task": "Train",
                        "state": "wait"
                    },
                    "TRAIN_SCORE": {
                        "current_task": "Score",
                        "state": "wait"
                    },
                    "TEST_SCORE": {
                        "current_task": "Score",
                        "state": "wait"
                    },
                },
            }
        }
