# -*- coding: utf-8 -*-

import os
import unittest
from StringIO import StringIO

from antlr3.tree import CommonTreeAdaptor, CommonTree, INVALID_TOKEN_TYPE
from antlr3.treewizard import TreeWizard, computeTokenTypes, \
     TreePatternLexer, EOF, ID, BEGIN, END, PERCENT, COLON, DOT, ARG, \
     TreePatternParser, \
     TreePattern, WildcardTreePattern, TreePatternTreeAdaptor


class TestComputeTokenTypes(unittest.TestCase):
    """Test case for the computeTokenTypes function."""

    def testNone(self):
        """computeTokenTypes(None) -> {}"""

        typeMap = computeTokenTypes(None)
        self.failUnless(isinstance(typeMap, dict))
        self.failUnlessEqual(typeMap, {})


    def testList(self):
        """computeTokenTypes(['a', 'b']) -> { 'a': 0, 'b': 1 }"""

        typeMap = computeTokenTypes(['a', 'b'])
        self.failUnless(isinstance(typeMap, dict))
        self.failUnlessEqual(typeMap, { 'a': 0, 'b': 1 })


class TestTreePatternLexer(unittest.TestCase):
    """Test case for the TreePatternLexer class."""

    def testBegin(self):
        """TreePatternLexer(): '('"""

        lexer = TreePatternLexer('(')
        type = lexer.nextToken()
        self.failUnlessEqual(type, BEGIN)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, False)

        
    def testEnd(self):
        """TreePatternLexer(): ')'"""

        lexer = TreePatternLexer(')')
        type = lexer.nextToken()
        self.failUnlessEqual(type, END)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, False)

        
    def testPercent(self):
        """TreePatternLexer(): '%'"""

        lexer = TreePatternLexer('%')
        type = lexer.nextToken()
        self.failUnlessEqual(type, PERCENT)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, False)

        
    def testDot(self):
        """TreePatternLexer(): '.'"""

        lexer = TreePatternLexer('.')
        type = lexer.nextToken()
        self.failUnlessEqual(type, DOT)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, False)

        
    def testColon(self):
        """TreePatternLexer(): ':'"""

        lexer = TreePatternLexer(':')
        type = lexer.nextToken()
        self.failUnlessEqual(type, COLON)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, False)

        
    def testEOF(self):
        """TreePatternLexer(): EOF"""

        lexer = TreePatternLexer('  \n \r \t ')
        type = lexer.nextToken()
        self.failUnlessEqual(type, EOF)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, False)

        
    def testID(self):
        """TreePatternLexer(): ID"""

        lexer = TreePatternLexer('_foo12_bar')
        type = lexer.nextToken()
        self.failUnlessEqual(type, ID)
        self.failUnlessEqual(lexer.sval, '_foo12_bar')
        self.failUnlessEqual(lexer.error, False)

        
    def testARG(self):
        """TreePatternLexer(): ARG"""

        lexer = TreePatternLexer('[ \\]bla\\n]')
        type = lexer.nextToken()
        self.failUnlessEqual(type, ARG)
        self.failUnlessEqual(lexer.sval, ' ]bla\\n')
        self.failUnlessEqual(lexer.error, False)


    def testError(self):
        """TreePatternLexer(): error"""

        lexer = TreePatternLexer('1')
        type = lexer.nextToken()
        self.failUnlessEqual(type, EOF)
        self.failUnlessEqual(lexer.sval, '')
        self.failUnlessEqual(lexer.error, True)


class TestTreePatternParser(unittest.TestCase):
    """Test case for the TreePatternParser class."""

    def setUp(self):
        """Setup text fixure

        We need a tree adaptor, use CommonTreeAdaptor.
        And a constant list of token names.
        
        """

        self.adaptor = CommonTreeAdaptor()
        self.tokens = [
            "", "", "", "", "", "A", "B", "C", "D", "E", "ID", "VAR"
            ]
        self.wizard = TreeWizard(self.adaptor, tokenNames=self.tokens)


    def testSingleNode(self):
        """TreePatternParser: 'ID'"""
        lexer = TreePatternLexer('ID')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(isinstance(tree, CommonTree))
        self.failUnlessEqual(tree.getType(), 10)
        self.failUnlessEqual(tree.getText(), 'ID')


    def testSingleNodeWithArg(self):
        """TreePatternParser: 'ID[foo]'"""
        lexer = TreePatternLexer('ID[foo]')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(isinstance(tree, CommonTree))
        self.failUnlessEqual(tree.getType(), 10)
        self.failUnlessEqual(tree.getText(), 'foo')


    def testSingleLevelTree(self):
        """TreePatternParser: '(A B)'"""
        lexer = TreePatternLexer('(A B)')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(isinstance(tree, CommonTree))
        self.failUnlessEqual(tree.getType(), 5)
        self.failUnlessEqual(tree.getText(), 'A')
        self.failUnlessEqual(tree.getChildCount(), 1)
        self.failUnlessEqual(tree.getChild(0).getType(), 6)
        self.failUnlessEqual(tree.getChild(0).getText(), 'B')
        

    def testNil(self):
        """TreePatternParser: 'nil'"""
        lexer = TreePatternLexer('nil')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(isinstance(tree, CommonTree))
        self.failUnlessEqual(tree.getType(), 0)
        self.failUnlessEqual(tree.getText(), None)
        

    def testWildcard(self):
        """TreePatternParser: '(.)'"""
        lexer = TreePatternLexer('(.)')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(isinstance(tree, WildcardTreePattern))
        

    def testLabel(self):
        """TreePatternParser: '(%a:A)'"""
        lexer = TreePatternLexer('(%a:A)')
        parser = TreePatternParser(lexer, self.wizard, TreePatternTreeAdaptor())
        tree = parser.pattern()
        self.failUnless(isinstance(tree, TreePattern))
        self.failUnlessEqual(tree.label, 'a')
        

    def testError1(self):
        """TreePatternParser: ')'"""
        lexer = TreePatternLexer(')')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(tree is None)
        

    def testError2(self):
        """TreePatternParser: '()'"""
        lexer = TreePatternLexer('()')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(tree is None)
        

    def testError3(self):
        """TreePatternParser: '(A ])'"""
        lexer = TreePatternLexer('(A ])')
        parser = TreePatternParser(lexer, self.wizard, self.adaptor)
        tree = parser.pattern()
        self.failUnless(tree is None)
        

class TestTreeWizard(unittest.TestCase):
    """Test case for the TreeWizard class."""

    def setUp(self):
        """Setup text fixure

        We need a tree adaptor, use CommonTreeAdaptor.
        And a constant list of token names.
        
        """

        self.adaptor = CommonTreeAdaptor()
        self.tokens = [
            "", "", "", "", "", "A", "B", "C", "D", "E", "ID", "VAR"
            ]


    def testInit(self):
        """TreeWizard.__init__()"""

        wiz = TreeWizard(
            self.adaptor,
            tokenNames=['a', 'b']
            )

        self.failUnless(wiz.adaptor is self.adaptor)
        self.failUnlessEqual(
            wiz.tokenNameToTypeMap,
            { 'a': 0, 'b': 1 }
            )


    def testGetTokenType(self):
        """TreeWizard.getTokenType()"""

        wiz = TreeWizard(
            self.adaptor,
            tokenNames=self.tokens
            )

        self.failUnlessEqual(
            wiz.getTokenType('A'),
            5
            )
            
        self.failUnlessEqual(
            wiz.getTokenType('VAR'),
            11
            )
            
        self.failUnlessEqual(
            wiz.getTokenType('invalid'),
            INVALID_TOKEN_TYPE
            )

    def testSingleNode(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("ID")
        found = t.toStringTree()
        expecting = "ID"
        self.failUnlessEqual(expecting, found)


    def testSingleNodeWithArg(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("ID[foo]")
        found = t.toStringTree()
        expecting = "foo"
        self.failUnlessEqual(expecting, found)


    def testSingleNodeTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A)")
        found = t.toStringTree()
        expecting = "A"
        self.failUnlessEqual(expecting, found)


    def testSingleLevelTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C D)")
        found = t.toStringTree()
        expecting = "(A B C D)"
        self.failUnlessEqual(expecting, found)


    def testListTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(nil A B C)")
        found = t.toStringTree()
        expecting = "A B C"
        self.failUnlessEqual(expecting, found)


    def testInvalidListTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("A B C")
        self.failUnless(t is None)


    def testDoubleLevelTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A (B C) (B D) E)")
        found = t.toStringTree()
        expecting = "(A (B C) (B D) E)"
        self.failUnlessEqual(expecting, found)


    def __simplifyIndexMap(self, indexMap):
        return dict( # stringify nodes for easy comparing
            (ttype, [str(node) for node in nodes])
            for ttype, nodes in indexMap.items()
            )
        
    def testSingleNodeIndex(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("ID")
        indexMap = wiz.index(tree)
        found = self.__simplifyIndexMap(indexMap)
        expecting = { 10: ["ID"] }
        self.failUnlessEqual(expecting, found)


    def testNoRepeatsIndex(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B C D)")
        indexMap = wiz.index(tree)
        found = self.__simplifyIndexMap(indexMap)
        expecting = { 8:['D'], 6:['B'], 7:['C'], 5:['A'] }
        self.failUnlessEqual(expecting, found)


    def testRepeatsIndex(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B (A C B) B D D)")
        indexMap = wiz.index(tree)
        found = self.__simplifyIndexMap(indexMap)
        expecting = { 8: ['D', 'D'], 6: ['B', 'B', 'B'], 7: ['C'], 5: ['A', 'A'] }
        self.failUnlessEqual(expecting, found)


    def testNoRepeatsVisit(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B C D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(str(node))
        
        wiz.visit(tree, wiz.getTokenType("B"), visitor)
        
        expecting = ['B']
        self.failUnlessEqual(expecting, elements)


    def testNoRepeatsVisit2(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B (A C B) B D D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(str(node))
        
        wiz.visit(tree, wiz.getTokenType("C"), visitor)
        
        expecting = ['C']
        self.failUnlessEqual(expecting, elements)


    def testRepeatsVisit(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B (A C B) B D D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(str(node))
        
        wiz.visit(tree, wiz.getTokenType("B"), visitor)
        
        expecting = ['B', 'B', 'B']
        self.failUnlessEqual(expecting, elements)


    def testRepeatsVisit2(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B (A C B) B D D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(str(node))
        
        wiz.visit(tree, wiz.getTokenType("A"), visitor)
        
        expecting = ['A', 'A']
        self.failUnlessEqual(expecting, elements)


    def testRepeatsVisitWithContext(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B (A C B) B D D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append('%s@%s[%d]' % (node, parent, childIndex))
        
        wiz.visit(tree, wiz.getTokenType("B"), visitor)
        
        expecting = ['B@A[0]', 'B@A[1]', 'B@A[2]']
        self.failUnlessEqual(expecting, elements)


    def testRepeatsVisitWithNullParentAndContext(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B (A C B) B D D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(
                '%s@%s[%d]'
                % (node, ['nil', parent][parent is not None], childIndex)
                )
        
        wiz.visit(tree, wiz.getTokenType("A"), visitor)
        
        expecting = ['A@nil[0]', 'A@A[1]']
        self.failUnlessEqual(expecting, elements)


    def testVisitPattern(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B C (A B) D)")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(
                str(node)
                )
        
        wiz.visit(tree, '(A B)', visitor)
        
        expecting = ['A'] # shouldn't match overall root, just (A B)
        self.failUnlessEqual(expecting, elements)


    def testVisitPatternMultiple(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B C (A B) (D (A B)))")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(
                '%s@%s[%d]'
                % (node, ['nil', parent][parent is not None], childIndex)
                )
        
        wiz.visit(tree, '(A B)', visitor)
        
        expecting = ['A@A[2]', 'A@D[0]']
        self.failUnlessEqual(expecting, elements)


    def testVisitPatternMultipleWithLabels(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        tree = wiz.create("(A B C (A[foo] B[bar]) (D (A[big] B[dog])))")

        elements = []
        def visitor(node, parent, childIndex, labels):
            elements.append(
                '%s@%s[%d]%s&%s'
                % (node,
                   ['nil', parent][parent is not None],
                   childIndex,
                   labels['a'],
                   labels['b'],
                   )
                )
        
        wiz.visit(tree, '(%a:A %b:B)', visitor)
        
        expecting = ['foo@A[2]foo&bar', 'big@D[0]big&dog']
        self.failUnlessEqual(expecting, elements)


    def testParse(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C)")
        valid = wiz.parse(t, "(A B C)")
        self.failUnless(valid)


    def testParseSingleNode(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("A")
        valid = wiz.parse(t, "A")
        self.failUnless(valid)


    def testParseSingleNodeFails(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("A")
        valid = wiz.parse(t, "B")
        self.failUnless(not valid)


    def testParseFlatTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(nil A B C)")
        valid = wiz.parse(t, "(nil A B C)")
        self.failUnless(valid)


    def testParseFlatTreeFails(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(nil A B C)")
        valid = wiz.parse(t, "(nil A B)")
        self.failUnless(not valid)


    def testParseFlatTreeFails2(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(nil A B C)")
        valid = wiz.parse(t, "(nil A B A)")
        self.failUnless(not valid)


    def testWildcard(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C)")
        valid = wiz.parse(t, "(A . .)")
        self.failUnless(valid)


    def testParseWithText(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B[foo] C[bar])")
        # C pattern has no text arg so despite [bar] in t, no need
        # to match text--check structure only.
        valid = wiz.parse(t, "(A B[foo] C)")
        self.failUnless(valid)


    def testParseWithTextFails(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C)")
        valid = wiz.parse(t, "(A[foo] B C)")
        self.failUnless(not valid) # fails


    def testParseLabels(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C)")
        labels = {}
        valid = wiz.parse(t, "(%a:A %b:B %c:C)", labels)
        self.failUnless(valid)
        self.failUnlessEqual("A", str(labels["a"]))
        self.failUnlessEqual("B", str(labels["b"]))
        self.failUnlessEqual("C", str(labels["c"]))


    def testParseWithWildcardLabels(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C)")
        labels = {}
        valid = wiz.parse(t, "(A %b:. %c:.)", labels)
        self.failUnless(valid)
        self.failUnlessEqual("B", str(labels["b"]))
        self.failUnlessEqual("C", str(labels["c"]))


    def testParseLabelsAndTestText(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B[foo] C)")
        labels = {}
        valid = wiz.parse(t, "(%a:A %b:B[foo] %c:C)", labels)
        self.failUnless(valid)
        self.failUnlessEqual("A", str(labels["a"]))
        self.failUnlessEqual("foo", str(labels["b"]))
        self.failUnlessEqual("C", str(labels["c"]))


    def testParseLabelsInNestedTree(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A (B C) (D E))")
        labels = {}
        valid = wiz.parse(t, "(%a:A (%b:B %c:C) (%d:D %e:E) )", labels)
        self.failUnless(valid)
        self.failUnlessEqual("A", str(labels["a"]))
        self.failUnlessEqual("B", str(labels["b"]))
        self.failUnlessEqual("C", str(labels["c"]))
        self.failUnlessEqual("D", str(labels["d"]))
        self.failUnlessEqual("E", str(labels["e"]))


    def testEquals(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t1 = wiz.create("(A B C)")
        t2 = wiz.create("(A B C)")
        same = wiz.equals(t1, t2)
        self.failUnless(same)


    def testEqualsWithText(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t1 = wiz.create("(A B[foo] C)")
        t2 = wiz.create("(A B[foo] C)")
        same = wiz.equals(t1, t2)
        self.failUnless(same)

	
    def testEqualsWithMismatchedText(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t1 = wiz.create("(A B[foo] C)")
        t2 = wiz.create("(A B C)")
        same = wiz.equals(t1, t2)
        self.failUnless(not same)


    def testEqualsWithMismatchedList(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t1 = wiz.create("(A B C)")
        t2 = wiz.create("(A B A)")
        same = wiz.equals(t1, t2)
        self.failUnless(not same)


    def testEqualsWithMismatchedListLength(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t1 = wiz.create("(A B C)")
        t2 = wiz.create("(A B)")
        same = wiz.equals(t1, t2)
        self.failUnless(not same)


    def testFindPattern(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C (A[foo] B[bar]) (D (A[big] B[dog])))")
        subtrees = wiz.find(t, "(A B)")
        found = [str(node) for node in subtrees]
        expecting = ['foo', 'big']
        self.failUnlessEqual(expecting, found)


    def testFindTokenType(self):
        wiz = TreeWizard(self.adaptor, self.tokens)
        t = wiz.create("(A B C (A[foo] B[bar]) (D (A[big] B[dog])))")
        subtrees = wiz.find(t, wiz.getTokenType('A'))
        found = [str(node) for node in subtrees]
        expecting = ['A', 'foo', 'big']
        self.failUnlessEqual(expecting, found)



if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
