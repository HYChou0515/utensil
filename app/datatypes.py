from __future__ import annotations

from typing import Union

from pydantic import BaseModel


class TNodeTask(BaseModel):
    name: str


class TNode(BaseModel):
    name: str
    receivers: list[Union[TNode, str]]
    callees: list[Union[TNode, str]]
    end_of_flow: bool
    switchon: bool
    tasks: list[TNodeTask]


TNode.update_forward_refs()


class TFlow(BaseModel):
    nodes: dict[str, TNode]
