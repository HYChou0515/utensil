from typing import Optional
import itertools
from fastapi import FastAPI, File, UploadFile

from core import save_flow, parse_flow
from datatypes import TNodeTask
from fastapi.middleware.cors import CORSMiddleware

from utensil.loopflow import loopflow
from utensil.loopflow.functions import basic, dataflow

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
async def query_parse_flow(file: UploadFile = File(...)):
    file_id = save_flow(file)
    flow_id, tflow = parse_flow(file_id)
    return {"flow_id": flow_id, "flow": tflow}
