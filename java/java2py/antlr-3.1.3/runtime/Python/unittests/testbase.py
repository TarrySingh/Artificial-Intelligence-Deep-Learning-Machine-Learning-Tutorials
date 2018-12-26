import unittest

class BrokenTest(unittest.TestCase.failureException):
    def __repr__(self):
        name, reason = self.args
        return '%s: %s: %s works now' % (
            (self.__class__.__name__, name, reason))


def broken(reason, *exceptions):
    '''Indicates a failing (or erroneous) test case fails that should succeed.
    If the test fails with an exception, list the exception type in args'''
    def wrapper(test_method):
        def replacement(*args, **kwargs):
            try:
                test_method(*args, **kwargs)
            except exceptions or unittest.TestCase.failureException:
                pass
            else:
                raise BrokenTest(test_method.__name__, reason)
        replacement.__doc__ = test_method.__doc__
        replacement.__name__ = 'XXX_' + test_method.__name__
        replacement.todo = reason
        return replacement
    return wrapper


