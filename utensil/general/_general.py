from typing import Any, Dict
from functools import partial

from _warnings import warn


def chunks(iterable, n):
    """Yield successive n-sized chunks from lst.

    >>> for chunk in chunks([1, 2, 3, 4, 5], 2):
    ...     print(chunk)
    [1, 2]
    [3, 4]
    [5]

    >>> for chunk in chunks(range(5), 2):
    ...     print(chunk)
    [0, 1]
    [2, 3]
    [4]

    >>> for chunk in chunks(range(5), 0):
    ...     print(chunk)

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
    """Warn if params have any items left.
    This method is useful for configuration parsing and warn if keys not used.

    >>> def parse_config(config):
    ...     first_name = config.pop('first_name', None)
    ...     last_name = config.pop('last_name', None)
    ...     warn_left_keys(config)
    >>> import warnings

    `config` is not well-defined, should be `last_name` instead of `lastname`.
    >>> config = {'first_name': 'Tom', 'lastname': 'Smith'}
    >>> with warnings.catch_warnings(record=True) as w:
    ...     parse_config(config)
    ...     warnings.simplefilter("always")
    ...     assert len(w) == 1
    ...     assert issubclass(w[-1].category, SyntaxWarning)
    ...     assert "'lastname' is not used" in str(w[-1].message)

    """
    for key in params.keys():
        warn(f"'{key}' is not used", category=SyntaxWarning)


open_utf8 = partial(open, encoding='UTF-8')
