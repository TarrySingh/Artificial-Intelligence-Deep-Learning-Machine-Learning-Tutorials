# module1.py
import sys

# don't do this!
sys.modules['test'] = lambda: 'Hello!'
