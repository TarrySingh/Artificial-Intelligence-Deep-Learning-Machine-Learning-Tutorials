# importer.py
print('Running importer.py')

import os.path
import types
import sys


def import_(module_name, module_file, module_path):
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_rel_file_path = os.path.join(module_path, module_file)
    module_abs_file_path = os.path.abspath(module_rel_file_path)

    # read source code from file
    with open(module_rel_file_path, 'r') as code_file:
        source_code = code_file.read()

    # next we create a module object
    mod = types.ModuleType(module_name)
    mod.__file__ = module_abs_file_path

    # insert a reference to the module in sys.modules
    sys.modules[module_name] = mod

    # compile the module source code into a code object
    # optionally we should tell the code object where the source came from
    # the third parameter is used to indicate that our source consists of a sequence of statements
    code = compile(source_code, filename=module_abs_file_path, mode='exec')

    # execute the module
    # we want the global variables to be stored in mod.__dict__
    exec(code, mod.__dict__)

    # return the module
    return sys.modules[module_name]
