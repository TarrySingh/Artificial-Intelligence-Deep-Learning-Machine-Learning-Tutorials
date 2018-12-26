#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is the default configuration file for java2python.  Unless
# explicity disabled with the '-n' or '--nodefaults' option, the j2py
# script will import this module for runtime configuration.

from java2python.mod import basic, transform
from java2python.lang.selector import *


# Leading indent character or characters.  Four spaces are used
# because that is the recommendation of PEP 8.
indentPrefix = ' ' * 4


# Prefix character or characters for comments.  The hash+space is
# recommended by PEP 8.
commentPrefix = '# '


# These values are strings or generators that yield strings
# for a module prologue.
modulePrologueHandlers = [
    basic.shebangLine,
    basic.simpleDocString,
    basic.maybeBsr,
    basic.maybeSyncHelpers,
]


# These generators yield lines for a module epilogue.
moduleEpilogueHandlers = [
    basic.scriptMainStanza,
]


# These generators yield (possibly modified) source strings for a
# module.  The `basic.outputSubs` handler references the list of
# regular expression substitutions near the end of this module.
moduleOutputHandlers = [
    basic.outputSubs,
]


# These generators yield doc strings for a class.
classHeadHandlers = [
    basic.simpleDocString,
]

methodParamHandlers = [
    basic.defaultParams,
]

# This is the name of the callable used to construct locks for an object with
# the synchronized keyword.
methodLockFunctionName = 'lock_for_object'

classBaseHandlers = [
    basic.defaultBases,
]

interfaceBaseHandlers = [
    basic.defaultBases,
]

# These generators are called after a class has been completely
# generated.  The class content sorter sorts the methods of a class by
# name.  It's commented out because its output differs so greatly
# from its input, and because it's really not very useful.
classPostWalkHandlers = [
    basic.moveStaticExpressions,
    ## basic.classContentSort,
]


enumHeadHandlers = [
    basic.simpleDocString,
]


interfaceHeadHandlers = [
    basic.simpleDocString,
    '__metaclass__ = ABCMeta',
]


interfacePostWalkMutators = [
]


methodHeadHandlers = [
    basic.simpleDocString,
]


methodPrologueHandlers = [
    basic.maybeAbstractMethod,
    basic.maybeClassMethod,
    # NB:  synchronized should come after classmethod
    basic.maybeSynchronizedMethod,
    basic.overloadedClassMethods,
]


# This handler creates enum values on enum classes after they've been
# defined.  The handler tries to match Java semantics by using
# strings.  Refer to the documentation for details.
enumValueHandler = basic.enumConstStrings

# Alternatively, you can use this handler to construct enum values as
# integers.
#enumValueHandler = basic.enumConstInts


# When the compiler needs to make up a variable name (for example, to
# emulate assignment expressions), it calls this handler to produce a
# new one.
expressionVariableNamingHandler = basic.globalNameCounter


# This handler simply creates comments in the file for package
# declarations.
modulePackageDeclarationHandler = basic.commentedPackages


# This handler can be used instead to create __init__.py files for
# 'namespace packages' via pkgutil.
# modulePackageDeclarationHandler = basic.namespacePackages


# This handler is turns java imports into python imports. No mapping
# of packages is performed:
moduleImportDeclarationHandler = basic.simpleImports

# This import decl. handler can be used instead to produce comments
# instead of import statements:
# moduleImportDeclarationHandler = basic.commentedImports

# The AST transformation function uses these declarations to modify an
# AST before compiling it to python source.  Having these declarations
# in a config file gives clients an opportunity to change the
# transformation behavior.

astTransforms = [
    (Type('NULL'),  transform.null2None),
    (Type('FALSE'), transform.false2False),
    (Type('TRUE'),  transform.true2True),
    (Type('IDENT'), transform.keywordSafeIdent),

    (Type('FLOATING_POINT_LITERAL'), transform.syntaxSafeFloatLiteral),

    (Type('TYPE') > Type('BOOLEAN'), transform.typeSub),
    (Type('TYPE') > Type('BYTE'), transform.typeSub),
    (Type('TYPE') > Type('CHAR'), transform.typeSub),
    (Type('TYPE') > Type('FLOAT'), transform.typeSub),
    (Type('TYPE') > Type('INT'), transform.typeSub),
    (Type('TYPE') > Type('SHORT'), transform.typeSub),
    (Type('TYPE') > Type('LONG'), transform.typeSub),
    (Type('TYPE') > Type('DOUBLE'), transform.typeSub),

    (Type('METHOD_CALL') > Type('DOT') > Type('IDENT', 'length'),
     transform.lengthToLen),

    (Type('TYPE') > Type('QUALIFIED_TYPE_IDENT') > Type('IDENT'),
     transform.typeSub),

]


# not implemented:

# minimum parameter count to trigger indentation of parameter names
# in method declarations.  set to 0 to disable.
#minIndentParams = 5

# Specifies handler for cast operations of non-primitive types are handled
# (primitive types are automatically handled).  Use basic.castDrop to leave
# cast expressions out of generated source.  Use basic.castCtor to transform
# casts into constructor calls.  Or you can specify a function of your own.
expressionCastHandler = basic.castDrop


# Values below are used by the handlers.  They're here for
# convenience.


# module output subs.
moduleOutputSubs = [
    (r'System\.out\.println\((.*)\)', r'print \1'),
    (r'System\.out\.print_\((.*?)\)', r'print \1,'),
    (r'(.*?)\.equals\((.*?)\)', r'\1 == \2'),
    (r'(.*?)\.equalsIgnoreCase\((.*?)\)', r'\1.lower() == \2.lower()'),
    (r'([\w.]+)\.size\(\)', r'len(\1)'),
    #(r'(\w+)\.get\((.*?)\)', r'\1[\2]'),
    (r'(\s)(\S*?)(\.toString\(\))', r'\1\2.__str__()'),
    (r'(\s)def toString', r'\1def __str__'),
    (r'(\s)(\S*?)(\.toLowerCase\(\))', r'\1\2.lower()'),
    (r'(.*?)IndexOutOfBoundsException\((.*?)\)', r'\1IndexError(\2)'),
    (r'\.__class__\.getName\(\)', '.__class__.__name__'),
    (r'\.getClass\(\)', '.__class__'),
    (r'\.getName\(\)', '.__name__'),
    (r'\.getInterfaces\(\)', '.__bases__'),
    #(r'String\.valueOf\((.*?)\)', r'str(\1)'),
    #(r'(\s)(\S*?)(\.toString\(\))', r'\1str(\2)'),
]


typeSubs = {
    'Boolean' : 'bool',
    'boolean' : 'bool',

    'Byte' : 'int',
    'byte' : 'int',

    'Char' : 'str',
    'char' : 'str',

    'Integer' : 'int',
    'int' : 'int',

    'Short' : 'int',
    'short' : 'int',

    'Long' : 'long',
    'long' : 'long',

    'Float' : 'float',
    'float' : 'float',

    'Double' : 'float',
    'double' : 'float',

    'String' : 'str',
    'java.lang.String' : 'str',

    'Object' : 'object',
    'IndexOutOfBoundsException' : 'IndexError',
    }
