import importlib
import os
import sys

args = sys.argv[1:]
if len(args) > 0:
    module = importlib.import_module(args[0])
    print(os.path.dirname(module.__file__))
