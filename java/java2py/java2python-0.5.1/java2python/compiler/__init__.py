#!/usr/bin/env python
# -*- coding: utf-8 -*-
# java2python.compiler package marker.
#
# This module provides a simpler facade over the rest of the compiler
# subpackage.  Client code should use the values in this module
# instead of using directly referencing items within the subpackage.

from java2python.compiler.block import Module
from java2python.lang import Lexer, Parser, StringStream, TokenStream, TreeAdaptor


def buildAST(source):
    """ Returns an AST for the given source. """
    lexer = Lexer(StringStream(source))
    parser = Parser(TokenStream(lexer))
    adapter = TreeAdaptor(lexer, parser)
    parser.setTreeAdaptor(adapter)
    scope = parser.javaSource()
    return scope.tree


def buildJavaDocAST(source):
    """ Returns an AST for the given javadoc source. """
    from java2python.lang.JavaDocLexer import JavaDocLexer
    from java2python.lang.JavaDocParser import JavaDocParser
    lexer = JavaDocLexer(StringStream(source))
    parser = JavaDocParser(TokenStream(lexer))
    scope = parser.commentBody()
    return scope.tree


def transformAST(tree, config):
    """ Walk the tree and apply the transforms in the config. """
    for selector, call in config.last('astTransforms', ()):
        for node in selector.walk(tree):
            call(node, config)
