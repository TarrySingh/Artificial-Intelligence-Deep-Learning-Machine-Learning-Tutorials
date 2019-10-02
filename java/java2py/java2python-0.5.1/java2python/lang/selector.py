#!/usr/bin/env python
# -*- coding: utf-8 -*-
# java2python.lang.selector -> declarative AST node selection
#
# This module provides classes for simple AST node selection that can be
# easily combined to form complex, declarative rules for retrieving AST
# nodes.
#
# The classes are similar to CSS selectors, with a nod to Python parsing
# libraries like LEPL and PyParsing.  At the moment, only a few very
# basic selector types are implemented, but those that are here already
# provide all of the functionality necessary for use within the package.
#
# Projects using java2python should regard this subpackage as
# experimental.  While the interfaces are not expected to change, the
# semantics may.  Use with caution.

from java2python.lang import tokens


class Selector(object):
    """ Base class for concrete selectors; provides operator methods. """

    def __add__(self, other):
        """ E + F

        Like CSS "E + F": an F element immediately preceded by an E element
        """
        return AdjacentSibling(self, other)

    def __and__(self, other):
        """ E & F

        Like CSS "E F":  an F element descendant of an E element
        """
        return Descendant(self, other)

    def __call__(self, *args, **kwds):
        """ Subclasses must implement. """
        raise NotImplementedError('Selector class cannot be called.')

    def __getitem__(self, key):
        """ E[n]

        Like CSS "E:nth-child(n)": an E element, the n-th child of its parent
        """
        return Nth(self, key)

    def __gt__(self, other):
        """ E > F

        Like CSS: "E > F": an F element child of an E element
        """
        return Child(self, other)

    def __div__(self, other):
        """ E / F

        Produces a AnySibling.
        """
        return AnySibling(self, other)

    def walk(self, tree):
        """ Select items from the tree and from the tree children. """
        for item in self(tree):
            yield item
        for child in tree.children:
            for item in self.walk(child):
                yield item


class Token(Selector):
    """ Token(...) -> select tokens by matching attributes.

    Token is the most generic and flexible Selector; using it,
    arbitrary nodes of any type, line number, position, and/or text
    can be selected.

    Calling Token() without any keywords is equivalent to:

        Token(channel=None, index=None, input=None, line=None,
              start=None, stop=None, text=None, type=None)

    Note that the value of each keyword can be a constant or a
    callable.  When callables are specified, they are passed a the
    token and should return a bool.
    """

    def __init__(self, **attrs):
        self.attrs = attrs
        ## we support strings so that the client can refer to the
        ## token name that way instead of via lookup or worse, integer.
        if isinstance(attrs.get('type'), (basestring, )):
            self.attrs['type'] = getattr(tokens, attrs.get('type'))

    def __call__(self, tree):
        items = self.attrs.items()
        token = tree.token

        def match_or_call(k, v):
            if callable(v):
                return v(token)
            return getattr(token, k)==v

        if all(match_or_call(k, v) for k, v in items if v is not None):
            yield tree

    def __str__(self):
        items = self.attrs.items()
        keys = ('{}={}'.format(k, v) for k, v in items if v is not None)
        return 'Token({})'.format(', '.join(keys))


class Nth(Selector):
    """ E[n] ->  match any slice n of E

    Similar to the :nth-child pseudo selector in CSS, but without the
    support for keywords like 'odd', 'even', etc.
    """
    def __init__(self, e, key):
        self.e, self.key = e, key

    def __call__(self, tree):
        for etree in self.e(tree):
            try:
                matches = tree.children[self.key]
            except (IndexError, ):
                return
            if not isinstance(matches, (list, )):
                matches = [matches]
            for child in matches:
                yield child

    def __str__(self):
        return 'Nth({0})[{1}]'.format(self.e, self.key)


class Child(Selector):
    """ E > F    select any F that is a child of E """

    def __init__(self, e, f):
        self.e, self.f = e, f

    def __call__(self, tree):
        for ftree in self.f(tree):
            for etree in self.e(tree.parent):
                yield ftree

    def __str__(self):
        return 'Child({0} > {1})'.format(self.e, self.f)


class Type(Selector):
    """ Type(T)    select any token of type T

    Similar to the type selector in CSS.
    """
    def __init__(self, key, value=None):
        self.key = key if isinstance(key, int) else getattr(tokens, key)
        self.value = value

    def __call__(self, tree):
        if tree.token.type == self.key:
            if self.value is None or self.value == tree.token.text:
                yield tree

    def __str__(self):
        val = '' if self.value is None else '={0}'.format(self.value)
        return 'Type({0}{1}:{2})'.format(tokens.map[self.key], val, self.key)


class Star(Selector):
    """ *    select any

    Similar to the * selector in CSS.
    """
    def __call__(self, tree):
        yield tree

    def __str__(self):
        return 'Star(*)'


class Descendant(Selector):
    """ E & F    select any F that is a descendant of E """

    def __init__(self, e, f):
        self.e, self.f = e, f

    def __call__(self, tree):
        for ftree in self.f(tree):
            root, ftree = ftree, ftree.parent
            while ftree:
                for etree in self.e(ftree):
                    yield root
                ftree = ftree.parent

    def __str__(self):
        return 'Descendant({0} & {1})'.format(self.e, self.f)


class AdjacentSibling(Selector):
    """ E + F    select any F immediately preceded by a sibling E """

    def __init__(self, e, f):
        self.e, self.f = e, f

    def __call__(self, node):
        if not node.parent:
            return
        for ftree in self.f(node):
            index = node.parent.children.index(ftree)
            if not index:
                return
            previous = node.parent.children[index-1]
            for child in self.e(previous):
                yield ftree

    def __str__(self):
        return 'AdjacentSibling({} + {})'.format(self.e, self.f)


class AnySibling(Selector):
    """ E / F    select any F preceded by a sibling E """

    def __init__(self, e, f):
        self.e, self.f = e, f

    def __call__(self, node):
        if not node.parent:
            return
        for ftree in self.f(node):
            index = node.parent.children.index(ftree)
            if not index:
                return
            for prev in node.parent.children[:index]:
                for child in self.e(prev):
                    yield ftree

    def __str__(self):
        return 'AnySibling({} / {})'.format(self.e, self.f)
