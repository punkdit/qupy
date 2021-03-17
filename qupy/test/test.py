#!/usr/bin/env python3

import sys
from collections.abc import Callable

# ----------------------------------------------------------------------------
# driver for the tests
#

class Skip(Exception):
    pass


def harvest(*modules):
    import traceback
    from qupy.test import test
    count, fails, skip = 0, 0, 0
    if sys.argv[1:]:
        match = sys.argv[1]
    else:
        match = ''
    funcs = []
    for module in modules:
        for name, value in list(module.__dict__.items()):
            if name.startswith('test') and isinstance(value, Callable) and match in name:
                funcs.append(value)
    funcs.sort(key = lambda f : (f.__module__, f.__code__.co_firstlineno))
    for func in funcs:
        print("%s.%s"%(func.__module__, func.__name__), end=' ')
        try:
            func()
            print(": OK")
        except AssertionError:
            print(": FAIL")
            traceback.print_exc()
            fails += 1
        except test.Skip:
            print(": SKIP")
            skip += 1
        except:
            print(": WAH")
            raise
        count += 1
    print("%d tests run, %d failed, %d skipped"%(count, fails, skip))

if __name__ == "__main__":
    from qupy.test import test_qu
    from qupy.test import test_diagrams
    from qupy.test import test_abstract
    harvest(test_abstract, test_qu, test_diagrams)


