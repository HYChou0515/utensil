from datetime import datetime

from pydantic import BaseModel


class FlowJobCreate(BaseModel):
    flow_id: str


class FlowJob(BaseModel):
    flow_id: str
    created_time: datetime


class FlowJobInDb(FlowJob):
    job_id: str
