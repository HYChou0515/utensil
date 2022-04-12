from typing import Optional, Any

from fastapi import Depends, FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware

from model import MFlow, MFlowJob, MFlowJobCreateByFlow
from model.flow_graph import MFlowGraph
from model.node_task import MNodeTaskListed
from service import FlowJobService, FlowService
from service.service import Service

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


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.get("/node-tasks", response_model=list[MNodeTaskListed])
async def query_list_node_tasks(service: Service = Depends()):
    return service.get_all_tasks()


@app.get("/task-source-code/{module}/{task_name}")
async def query_get_source_code_of_task(module: str,
                                        task_name: str,
                                        service: Service = Depends()):
    return service.get_source_code_of_node_task(module, task_name)


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


@app.post("/graph")
async def create_graph(graph_body: MFlowGraph, service: Service = Depends()):
    return service.create_graph(graph_body)
