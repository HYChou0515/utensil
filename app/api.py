import itertools
from typing import Optional

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utensil.loopflow import loopflow
from utensil.loopflow.functions import basic, dataflow

from model import MFlow, MFlowJob, MFlowJobCreateByFlow, MNodeTask
from service import FlowJobService, FlowService

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


@app.get("/node-tasks", response_model=list[MNodeTask])
async def query_list_node_tasks():
    return [
        MNodeTask(name=name)
        for name, _ in itertools.chain(basic_tasks, dataflow_tasks)
    ]


@app.post("/parse-flow", response_model=MFlow)
async def query_parse_flow(file: UploadFile = File(...),
                           flow_service: FlowService = Depends(FlowService)):
    flow_file_in_db = await flow_service.save_flow(file)
    flow_in_db = await flow_service.parse_flow(flow_file_in_db.id)
    return flow_in_db


@app.post("/flow/job", response_model=MFlowJob)
async def create_flow_job(
        flow_job_create: MFlowJobCreateByFlow,
        flow_job_service: FlowJobService = Depends(FlowJobService),
):
    return await flow_job_service.create_flow_job(flow_job_create)
