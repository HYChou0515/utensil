from typing import List, Tuple, Literal, Dict, Union, Any, Optional

from pydantic import Field

from model import Model

GraphObjectId = str


class MFlowGraphLink(Model):
    id: GraphObjectId
    source: GraphObjectId
    sourcePort: GraphObjectId
    target: GraphObjectId
    targetPort: GraphObjectId


class MFlowGraphPort(Model):
    id: GraphObjectId
    isIn: bool = Field(alias="in")
    name: str
    parentNode: GraphObjectId
    links: List[GraphObjectId]


class MFlowGraphNode(Model):
    id: GraphObjectId
    nodeType: Literal['task', 'switch-on', 'end-of-flow']
    module: Optional[str]
    params: List[Tuple[str, Literal['required', 'optional']]]
    ports: List[MFlowGraphPort]


class MFlowGraphModel(Model):
    models: Dict[GraphObjectId, Union[MFlowGraphLink, MFlowGraphNode]]


class MFlowGraph(Model):
    layers: List[MFlowGraphModel]
