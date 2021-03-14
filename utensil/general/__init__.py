

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
