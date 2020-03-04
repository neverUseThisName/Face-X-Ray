import os, errno, sys, shutil
from os.path import join, splitext, join, basename

def mkdir_p(path):
    try:
        os.makedirs(os.path.abspath(path))
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def files(path, exts=None, r=False):
    if os.path.isfile(path):
        if exts is None or (exts is not None and splitext(path)[-1] in exts):
            yield path
    elif os.path.isdir(path):
        for p, _, fs in os.walk(path):
            for f in sorted(fs):
                if exts is not None:
                    if splitext(f)[1] in exts:
                        yield join(p, f)
                else:
                    yield join(p, f)
            if not r:
                break


if __name__ == "__main__":
    pass