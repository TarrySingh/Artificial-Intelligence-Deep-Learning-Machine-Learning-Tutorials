#!/usr/bin/env python
# -*- coding: utf-8 -*-
# java2python.mod.basic -> functions to revise generated source strings.

from itertools import count
from logging import info, warn
from os import path
from re import sub as rxsub

from java2python.lib import FS


def shebangLine(module):
    """ yields the canonical python shebang line. """
    yield '#!/usr/bin/env python'


def encodingLine(encoding='utf-8'):
    """ returns a function to yield the specified encoding line.

    Note that this function isn't wired up because the encoding is
    specified for the source directly, and adding this line produces a
    syntax error when the compile function is used.
    """
    def line(module):
        yield '# -*- coding: {0} -*-'.format(encoding)
    return line


def simpleDocString(obj):
    """ yields multiple lines for a default docstring.

    This generator works for modules, classes, and functions.
    """
    yield '""" generated source for {0} {1} """'.format(obj.typeName, obj.name)


def commentedImports(module, expr):
    module.factory.comment(parent=module, left=expr, fs='import: {left}')


def simpleImports(module, expr):
    module.factory.expr(parent=module, left=expr, fs='import {left}')


def commentedPackages(module, expr):
    module.factory.comment(parent=module, left=expr, fs='# package: {left}')


def namespacePackages(module, expr):
    source = module.sourceFilename
    if not source:
        warn('namespace package not created; source input not named.')
        return
    initname = path.join(path.dirname(source), '__init__.py')
    if path.exists(initname):
        warn('namespace package not created; __init__.py exists.')
        return
    with open(initname, 'w') as initfile:
        initfile.write('from pkgutil import extend_path\n')
        initfile.write('__path__ = extend_path(__path__, __name__)\n')
        # wrong
        initfile.write('\nfrom {0} import {0}\n'.format(module.name))
    info('created __init__.py file for package %s.', expr)


def enumConstInts(enum, index, name):
    return str(index)


def enumConstStrings(enum, index, name):
    return repr(name)


scriptTemplate = """\n
if __name__ == '__main__':
{indent}import sys
{indent}{name}.main(sys.argv)"""


def scriptMainStanza(module):
    def filterClass(x):
        return x.isClass and x.name==module.name

    def filterMethod(x):
        return x.isMethod and x.isPublic and x.isStatic and \
               x.isVoid and x.name=='main'

    for cls in [c for c in module.children if filterClass(c)]:
        if [m for m in cls.children if filterMethod(m)]:
            yield scriptTemplate.format(indent=module.indent, name=module.name)
            break


def outputSubs(obj, text):
    subsname = '{0}OutputSubs'.format(obj.typeName)
    subs = obj.config.every(subsname, [])
    for sub in subs:
        for pattern, repl in sub:
            text = rxsub(pattern, repl, text)
    return text


def overloadedClassMethods(method):
    """
    NB: this implementation does not handle overloaded static (or
    class) methods, only instance methods.
    """
    cls = method.parent
    methods = [o for o in cls.children if o.isMethod and o.name==method.name]
    if len(methods) == 1:
        return
    for i, m in enumerate(methods[1:]):
        args = [p['type'] for p in m.parameters]
        args = ', '.join(args)
        m.decorators.append('@{0}.register({1})'.format(method.name, args))
        m.name = '{0}_{1}'.format(method.name, i)
    # for this one only:
    yield '@overloaded'


def maybeClassMethod(method):
    if method.isStatic and 'classmethod' not in method.decorators:
        yield '@classmethod'


def maybeAbstractMethod(method):
    if method.parent and method.parent.isInterface:
        yield '@abstractmethod'


def maybeSynchronizedMethod(method):
    if 'synchronized' in method.modifiers:
        module = method.parents(lambda x:x.isModule).next()
        module.needsSyncHelpers = True
        yield '@synchronized'


def globalNameCounter(original, counter=count()):
    return '__{0}_{1}'.format(original, counter.next())


def getBsrSrc():
    from inspect import getsource
    from java2python.mod.include.bsr import bsr
    return getsource(bsr)


def getSyncHelpersSrc():
    from inspect import getsource
    from java2python.mod.include import sync
    return getsource(sync)


def maybeBsr(module):
    if getattr(module, 'needsBsrFunc', False):
        for line in getBsrSrc().split('\n'):
            yield line


def maybeSyncHelpers(module):
    if getattr(module, 'needsSyncHelpers', False):
        for line in getSyncHelpersSrc().split('\n'):
            yield line


def classContentSort(obj):
    isMethod = lambda x:x and x.isMethod

    def iterBody(body):
        group = []
        for value in body:
            if isMethod(value):
                group.append(value)
                yield group
                group = []
            else:
                group.append(value)
        yield group

    def sortBody(group):
        methods = [item for item in group if isMethod(item)]
        return methods[0].name if methods else -1

    grp = list(iterBody(obj.children))
    grpsrt = sorted(grp, key=sortBody)
    obj.children = [item for grp in grpsrt for item in grp]


def defaultParams(obj):
    return iter(obj.parameters)


def zopeInterfaceMethodParams(obj):
    if not obj.parent.isInterface:
        for param in obj.parameters:
            yield param
    else:
        for index, param in enumerate(obj.parameters):
            if index != 0 and param['name'] != 'self':
                yield param


normalBases = ('object', )


def defaultBases(obj):
    return iter(obj.bases or normalBases)


def zopeInterfaceBases(obj):
    return iter(obj.bases or ['zope.interface.Interface'])


def implAny(obj):
    for module in obj.parents(lambda x:x.isModule):
        for name in obj.bases:
            if any(module.find(lambda v:v.name == name)):
                return True


def zopeImplementsClassBases(obj):
    return iter(normalBases) if implAny(obj) else defaultBases(obj)


def zopeImplementsClassHead(obj):
    if implAny(obj):
        for cls in obj.bases:
            yield 'zope.interface.implements({})'.format(cls)


def moveStaticExpressions(cls):
    name = '{}.'.format(cls.name) # notice the dot
    exprs = [child for child in cls.children if child.isExpression and name in str(child)]
    module = cls.parents(lambda x:x.isModule).next()
    for expr in exprs:
        cls.children.remove(expr)
        newExpr = module.factory.expr(fs=name + '{right}', right=expr)
        module.adopt(newExpr, index=len(module.children))


def castCtor(expr, node):
    expr.fs = FS.l + '(' + FS.r + ')'


def castDrop(expr, node):
    expr.fs = FS.r
