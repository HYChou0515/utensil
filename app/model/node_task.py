from typing import List
from model import Model


class MNodeTaskListed(Model):
    key: str
    module: str
    task_name: str
    arg_names: List[str]
