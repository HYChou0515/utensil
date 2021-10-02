from typing import Any, Dict
from functools import partial

from _warnings import warn


def chunks(iterable, n):
    """
    Yield successive n-sized chunks from lst.
    """
    if n == 0:
        yield from []
    elif isinstance(iterable, (list, str, tuple)):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]
    else:
        ret = []
        for it in iterable:
            ret.append(it)
            if len(ret) == n:
                yield ret
                ret = []
        if len(ret) > 0:
            yield ret


def warn_left_keys(params: Dict[str, Any]):
    for key in params.keys():
        warn(f"{key} is not used", category=SyntaxWarning)


open_utf8 = partial(open, encoding='UTF-8')
