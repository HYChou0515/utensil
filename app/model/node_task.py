from typing import List, Tuple, Literal
from model import Model


class MNodeTaskListed(Model):
    key: str
    module: str
    task_name: str
    arg_names: List[str]
    params: List[Tuple[str, Literal['required', 'optional']]]
