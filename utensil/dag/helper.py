from _warnings import warn
from typing import Dict, Any


def warn_left_keys(params: Dict[str, Any]):
    for key in params.keys():
        warn(f'{key} is not used', category=SyntaxWarning)
