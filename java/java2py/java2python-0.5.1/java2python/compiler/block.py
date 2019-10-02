#!/usr/bin/env python
# -*- coding: utf-8 -*-
# java2python.compiler.block -> Visitors combined with templates.
#
# This module defines classes which combine AST walking with source
# generation.  We've put these two behaviors into separate modules,
# java2python.compiler.template for creating source code, and
# java2python.compiler.visitor for walking ANTLR trees.
#
# Each of the classes depends on the behavior of its counterpart.
# This means they're very tightly coupled and that the classes are not
# very reusable.  The module split does allow for grouping of related
# methods and does hide the cluttered code.

from sys import modules
from java2python.compiler import template, visitor


def addTypeToModule((className, factoryName)):
    """ Constructs and adds a new type to this module. """
    bases = (getattr(template, className), getattr(visitor, className))
    newType = type(className, bases, dict(factoryName=factoryName))
    setattr(modules[__name__], className, newType)


map(addTypeToModule, (
        ('Annotation',    'at'),
        ('Class',         'klass'),
        ('Comment',       'comment'),
        ('Enum',          'enum'),
        ('Expression',    'expr'),
        ('Interface',     'interface'),
        ('Method',        'method'),
        ('MethodContent', 'methodContent'),
        ('Module',        'module'),
        ('Statement',     'statement'),
        )
    )
