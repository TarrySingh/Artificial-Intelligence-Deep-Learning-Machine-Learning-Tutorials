# main.py
import os.path
import types
import sys

# let's "import" module1 manually

# first we need to load the code from file
module_name = 'module1'
module_file = 'module1_source.py'
module_path = '.'

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

# our module is now imported!
# We can use it directly via our mod variable

mod.hello()

# but we can also import it, using the module name we specified
import module1

module1.hello()




