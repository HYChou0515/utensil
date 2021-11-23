import itertools
from typing import Optional

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utensil.loopflow import loopflow
from utensil.loopflow.functions import basic, dataflow

from schema.datatypes import TNodeTask
from schema.flow_job import FlowJobCreate, FlowJobInDb
from service.core import FlowJobService, FlowService

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def all_tasks(task_module):
    for task in task_module.__dict__.values():
        if (isinstance(task, type) and task is not loopflow.NodeTask and
                issubclass(task, loopflow.NodeTask)):
            name = loopflow.default_node_task_name(task)
            yield name, task


basic_tasks = list(all_tasks(basic))
dataflow_tasks = list(all_tasks(dataflow))


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/node-tasks", response_model=list[TNodeTask])
async def query_list_node_tasks():
    return [
        TNodeTask(name=name)
        for name, _ in itertools.chain(basic_tasks, dataflow_tasks)
    ]


@app.post("/parse-flow")
async def query_parse_flow(file: UploadFile = File(...),
                           flow_service: FlowService = Depends(FlowService)):
    file_id = flow_service.save_flow(file)
    flow_id, tflow = flow_service.parse_flow(file_id)
    return {"flow_id": flow_id, "flow": tflow}


@app.post("/flow/job", response_model=FlowJobInDb)
async def create_flow_job(
        flow_job_create: FlowJobCreate,
        flow_job_service: FlowJobService = Depends(FlowJobService),
):
    return flow_job_service.create_flow_job(flow_job_create)
