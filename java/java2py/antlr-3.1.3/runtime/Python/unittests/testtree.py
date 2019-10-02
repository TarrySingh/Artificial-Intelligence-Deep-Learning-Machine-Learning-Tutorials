# -*- coding: utf-8 -*-

import os
import unittest
from StringIO import StringIO

from antlr3.tree import (CommonTreeNodeStream, CommonTree, CommonTreeAdaptor,
                         TreeParser, TreeVisitor)
from antlr3 import CommonToken, UP, DOWN, EOF
from antlr3.treewizard import TreeWizard

class TestTreeNodeStream(unittest.TestCase):
    """Test case for the TreeNodeStream class."""

    def setUp(self):
        self.adaptor = CommonTreeAdaptor()


    def newStream(self, t):
        """Build new stream; let's us override to test other streams."""
        return CommonTreeNodeStream(t)


    def testSingleNode(self):
        t = CommonTree(CommonToken(101))

        stream = self.newStream(t)
        expecting = "101"
        found = self.toNodesOnlyString(stream)
        self.failUnlessEqual(expecting, found)

        expecting = "101"
        found = str(stream)
        self.failUnlessEqual(expecting, found)


    def testTwoChildrenOfNilRoot(self):
        class V(CommonTree):
            def __init__(self, token=None, ttype=None, x=None):
                if x is not None:
                    self.x = x

                if ttype is not None and token is None:
                    self.token = CommonToken(type=ttype)

                if token is not None:
                    self.token = token

            def __str__(self):
                if self.token is not None:
                    txt = self.token.text
                else:
                    txt = ""

                txt += "<V>"
                return txt

        root_0 = self.adaptor.nil();
        t = V(ttype=101, x=2)
        u = V(token=CommonToken(type=102, text="102"))
        self.adaptor.addChild(root_0, t)
        self.adaptor.addChild(root_0, u)
        self.assert_(root_0.parent is None)
        self.assertEquals(-1, root_0.childIndex)
        self.assertEquals(0, t.childIndex)
        self.assertEquals(1, u.childIndex)


    def test4Nodes(self):
        # ^(101 ^(102 103) 104)
        t = CommonTree(CommonToken(101))
        t.addChild(CommonTree(CommonToken(102)))
        t.getChild(0).addChild(CommonTree(CommonToken(103)))
        t.addChild(CommonTree(CommonToken(104)))

        stream = self.newStream(t)
        expecting = "101 102 103 104"
        found = self.toNodesOnlyString(stream)
        self.failUnlessEqual(expecting, found)

        expecting = "101 2 102 2 103 3 104 3"
        found = str(stream)
        self.failUnlessEqual(expecting, found)


    def testList(self):
        root = CommonTree(None)

        t = CommonTree(CommonToken(101))
        t.addChild(CommonTree(CommonToken(102)))
        t.getChild(0).addChild(CommonTree(CommonToken(103)))
        t.addChild(CommonTree(CommonToken(104)))

        u = CommonTree(CommonToken(105))

        root.addChild(t)
        root.addChild(u)

        stream = CommonTreeNodeStream(root)
        expecting = "101 102 103 104 105"
        found = self.toNodesOnlyString(stream)
        self.failUnlessEqual(expecting, found)

        expecting = "101 2 102 2 103 3 104 3 105"
        found = str(stream)
        self.failUnlessEqual(expecting, found)


    def testFlatList(self):
        root = CommonTree(None)

        root.addChild(CommonTree(CommonToken(101)))
        root.addChild(CommonTree(CommonToken(102)))
        root.addChild(CommonTree(CommonToken(103)))

        stream = CommonTreeNodeStream(root)
        expecting = "101 102 103"
        found = self.toNodesOnlyString(stream)
        self.failUnlessEqual(expecting, found)

        expecting = "101 102 103"
        found = str(stream)
        self.failUnlessEqual(expecting, found)


    def testListWithOneNode(self):
        root = CommonTree(None)

        root.addChild(CommonTree(CommonToken(101)))

        stream = CommonTreeNodeStream(root)
        expecting = "101"
        found = self.toNodesOnlyString(stream)
        self.failUnlessEqual(expecting, found)

        expecting = "101"
        found = str(stream)
        self.failUnlessEqual(expecting, found)


    def testAoverB(self):
        t = CommonTree(CommonToken(101))
        t.addChild(CommonTree(CommonToken(102)))

        stream = self.newStream(t)
        expecting = "101 102"
        found = self.toNodesOnlyString(stream)
        self.failUnlessEqual(expecting, found)

        expecting = "101 2 102 3"
        found = str(stream)
        self.failUnlessEqual(expecting, found)


    def testLT(self):
        # ^(101 ^(102 103) 104)
        t = CommonTree(CommonToken(101))
        t.addChild(CommonTree(CommonToken(102)))
        t.getChild(0).addChild(CommonTree(CommonToken(103)))
        t.addChild(CommonTree(CommonToken(104)))

        stream = self.newStream(t)
        self.failUnlessEqual(101, stream.LT(1).getType())
        self.failUnlessEqual(DOWN, stream.LT(2).getType())
        self.failUnlessEqual(102, stream.LT(3).getType())
        self.failUnlessEqual(DOWN, stream.LT(4).getType())
        self.failUnlessEqual(103, stream.LT(5).getType())
        self.failUnlessEqual(UP, stream.LT(6).getType())
        self.failUnlessEqual(104, stream.LT(7).getType())
        self.failUnlessEqual(UP, stream.LT(8).getType())
        self.failUnlessEqual(EOF, stream.LT(9).getType())
        # check way ahead
        self.failUnlessEqual(EOF, stream.LT(100).getType())


    def testMarkRewindEntire(self):
        # ^(101 ^(102 103 ^(106 107) ) 104 105)
        # stream has 7 real + 6 nav nodes
        # Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r0.addChild(r1)
        r1.addChild(CommonTree(CommonToken(103)))
        r2 = CommonTree(CommonToken(106))
        r2.addChild(CommonTree(CommonToken(107)))
        r1.addChild(r2)
        r0.addChild(CommonTree(CommonToken(104)))
        r0.addChild(CommonTree(CommonToken(105)))

        stream = CommonTreeNodeStream(r0)
        m = stream.mark() # MARK
        for _ in range(13): # consume til end
            stream.LT(1)
            stream.consume()

        self.failUnlessEqual(EOF, stream.LT(1).getType())
        self.failUnlessEqual(UP, stream.LT(-1).getType())
        stream.rewind(m)      # REWIND

        # consume til end again :)
        for _ in range(13): # consume til end
            stream.LT(1)
            stream.consume()

        self.failUnlessEqual(EOF, stream.LT(1).getType())
        self.failUnlessEqual(UP, stream.LT(-1).getType())


    def testMarkRewindInMiddle(self):
        # ^(101 ^(102 103 ^(106 107) ) 104 105)
        # stream has 7 real + 6 nav nodes
        # Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r0.addChild(r1)
        r1.addChild(CommonTree(CommonToken(103)))
        r2 = CommonTree(CommonToken(106))
        r2.addChild(CommonTree(CommonToken(107)))
        r1.addChild(r2)
        r0.addChild(CommonTree(CommonToken(104)))
        r0.addChild(CommonTree(CommonToken(105)))

        stream = CommonTreeNodeStream(r0)
        for _ in range(7): # consume til middle
            #System.out.println(tream.LT(1).getType())
            stream.consume()

        self.failUnlessEqual(107, stream.LT(1).getType())
        m = stream.mark() # MARK
        stream.consume() # consume 107
        stream.consume() # consume UP
        stream.consume() # consume UP
        stream.consume() # consume 104
        stream.rewind(m)      # REWIND

        self.failUnlessEqual(107, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(UP, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(UP, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(104, stream.LT(1).getType())
        stream.consume()
        # now we're past rewind position
        self.failUnlessEqual(105, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(UP, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(EOF, stream.LT(1).getType())
        self.failUnlessEqual(UP, stream.LT(-1).getType())


    def testMarkRewindNested(self):
        # ^(101 ^(102 103 ^(106 107) ) 104 105)
        # stream has 7 real + 6 nav nodes
        # Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r0.addChild(r1)
        r1.addChild(CommonTree(CommonToken(103)))
        r2 = CommonTree(CommonToken(106))
        r2.addChild(CommonTree(CommonToken(107)))
        r1.addChild(r2)
        r0.addChild(CommonTree(CommonToken(104)))
        r0.addChild(CommonTree(CommonToken(105)))

        stream = CommonTreeNodeStream(r0)
        m = stream.mark() # MARK at start
        stream.consume() # consume 101
        stream.consume() # consume DN
        m2 = stream.mark() # MARK on 102
        stream.consume() # consume 102
        stream.consume() # consume DN
        stream.consume() # consume 103
        stream.consume() # consume 106
        stream.rewind(m2)      # REWIND to 102
        self.failUnlessEqual(102, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume()
        # stop at 103 and rewind to start
        stream.rewind(m) # REWIND to 101
        self.failUnlessEqual(101, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(102, stream.LT(1).getType())
        stream.consume()
        self.failUnlessEqual(DOWN, stream.LT(1).getType())


    def testSeek(self):
        # ^(101 ^(102 103 ^(106 107) ) 104 105)
        # stream has 7 real + 6 nav nodes
        # Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r0.addChild(r1)
        r1.addChild(CommonTree(CommonToken(103)))
        r2 = CommonTree(CommonToken(106))
        r2.addChild(CommonTree(CommonToken(107)))
        r1.addChild(r2)
        r0.addChild(CommonTree(CommonToken(104)))
        r0.addChild(CommonTree(CommonToken(105)))

        stream = CommonTreeNodeStream(r0)
        stream.consume() # consume 101
        stream.consume() # consume DN
        stream.consume() # consume 102
        stream.seek(7)   # seek to 107
        self.failUnlessEqual(107, stream.LT(1).getType())
        stream.consume() # consume 107
        stream.consume() # consume UP
        stream.consume() # consume UP
        self.failUnlessEqual(104, stream.LT(1).getType())


    def testSeekFromStart(self):
        # ^(101 ^(102 103 ^(106 107) ) 104 105)
        # stream has 7 real + 6 nav nodes
        # Sequence of types: 101 DN 102 DN 103 106 DN 107 UP UP 104 105 UP EOF
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r0.addChild(r1)
        r1.addChild(CommonTree(CommonToken(103)))
        r2 = CommonTree(CommonToken(106))
        r2.addChild(CommonTree(CommonToken(107)))
        r1.addChild(r2)
        r0.addChild(CommonTree(CommonToken(104)))
        r0.addChild(CommonTree(CommonToken(105)))

        stream = CommonTreeNodeStream(r0)
        stream.seek(7)   # seek to 107
        self.failUnlessEqual(107, stream.LT(1).getType())
        stream.consume() # consume 107
        stream.consume() # consume UP
        stream.consume() # consume UP
        self.failUnlessEqual(104, stream.LT(1).getType())


    def toNodesOnlyString(self, nodes):
        buf = []
        for i in range(nodes.size()):
            t = nodes.LT(i+1)
            type = nodes.getTreeAdaptor().getType(t)
            if not (type==DOWN or type==UP):
                buf.append(str(type))

        return ' '.join(buf)
    

class TestCommonTreeNodeStream(unittest.TestCase):
    """Test case for the CommonTreeNodeStream class."""

    def testPushPop(self):
        # ^(101 ^(102 103) ^(104 105) ^(106 107) 108 109)
        # stream has 9 real + 8 nav nodes
        # Sequence of types: 101 DN 102 DN 103 UP 104 DN 105 UP 106 DN 107 UP 108 109 UP
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r1.addChild(CommonTree(CommonToken(103)))
        r0.addChild(r1)
        r2 = CommonTree(CommonToken(104))
        r2.addChild(CommonTree(CommonToken(105)))
        r0.addChild(r2)
        r3 = CommonTree(CommonToken(106))
        r3.addChild(CommonTree(CommonToken(107)))
        r0.addChild(r3)
        r0.addChild(CommonTree(CommonToken(108)))
        r0.addChild(CommonTree(CommonToken(109)))

        stream = CommonTreeNodeStream(r0)
        expecting = "101 2 102 2 103 3 104 2 105 3 106 2 107 3 108 109 3"
        found = str(stream)
        self.failUnlessEqual(expecting, found)

        # Assume we want to hit node 107 and then "call 102" then return

        indexOf102 = 2
        indexOf107 = 12
        for _ in range(indexOf107):# consume til 107 node
            stream.consume()
        
        # CALL 102
        self.failUnlessEqual(107, stream.LT(1).getType())
        stream.push(indexOf102)
        self.failUnlessEqual(102, stream.LT(1).getType())
        stream.consume() # consume 102
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume() # consume DN
        self.failUnlessEqual(103, stream.LT(1).getType())
        stream.consume() # consume 103
        self.failUnlessEqual(UP, stream.LT(1).getType())
        # RETURN
        stream.pop()
        self.failUnlessEqual(107, stream.LT(1).getType())


    def testNestedPushPop(self):
        # ^(101 ^(102 103) ^(104 105) ^(106 107) 108 109)
        # stream has 9 real + 8 nav nodes
        # Sequence of types: 101 DN 102 DN 103 UP 104 DN 105 UP 106 DN 107 UP 108 109 UP
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r1.addChild(CommonTree(CommonToken(103)))
        r0.addChild(r1)
        r2 = CommonTree(CommonToken(104))
        r2.addChild(CommonTree(CommonToken(105)))
        r0.addChild(r2)
        r3 = CommonTree(CommonToken(106))
        r3.addChild(CommonTree(CommonToken(107)))
        r0.addChild(r3)
        r0.addChild(CommonTree(CommonToken(108)))
        r0.addChild(CommonTree(CommonToken(109)))

        stream = CommonTreeNodeStream(r0)

        # Assume we want to hit node 107 and then "call 102", which
        # calls 104, then return

        indexOf102 = 2
        indexOf107 = 12
        for _ in range(indexOf107): # consume til 107 node
            stream.consume()

        self.failUnlessEqual(107, stream.LT(1).getType())
        # CALL 102
        stream.push(indexOf102)
        self.failUnlessEqual(102, stream.LT(1).getType())
        stream.consume() # consume 102
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume() # consume DN
        self.failUnlessEqual(103, stream.LT(1).getType())
        stream.consume() # consume 103

        # CALL 104
        indexOf104 = 6
        stream.push(indexOf104)
        self.failUnlessEqual(104, stream.LT(1).getType())
        stream.consume() # consume 102
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume() # consume DN
        self.failUnlessEqual(105, stream.LT(1).getType())
        stream.consume() # consume 103
        self.failUnlessEqual(UP, stream.LT(1).getType())
        # RETURN (to UP node in 102 subtree)
        stream.pop()

        self.failUnlessEqual(UP, stream.LT(1).getType())
        # RETURN (to empty stack)
        stream.pop()
        self.failUnlessEqual(107, stream.LT(1).getType())


    def testPushPopFromEOF(self):
        # ^(101 ^(102 103) ^(104 105) ^(106 107) 108 109)
        # stream has 9 real + 8 nav nodes
        # Sequence of types: 101 DN 102 DN 103 UP 104 DN 105 UP 106 DN 107 UP 108 109 UP
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r1.addChild(CommonTree(CommonToken(103)))
        r0.addChild(r1)
        r2 = CommonTree(CommonToken(104))
        r2.addChild(CommonTree(CommonToken(105)))
        r0.addChild(r2)
        r3 = CommonTree(CommonToken(106))
        r3.addChild(CommonTree(CommonToken(107)))
        r0.addChild(r3)
        r0.addChild(CommonTree(CommonToken(108)))
        r0.addChild(CommonTree(CommonToken(109)))

        stream = CommonTreeNodeStream(r0)

        while stream.LA(1) != EOF:
            stream.consume()

        indexOf102 = 2
        indexOf104 = 6
        self.failUnlessEqual(EOF, stream.LT(1).getType())

        # CALL 102
        stream.push(indexOf102)
        self.failUnlessEqual(102, stream.LT(1).getType())
        stream.consume() # consume 102
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume() # consume DN
        self.failUnlessEqual(103, stream.LT(1).getType())
        stream.consume() # consume 103
        self.failUnlessEqual(UP, stream.LT(1).getType())
        # RETURN (to empty stack)
        stream.pop()
        self.failUnlessEqual(EOF, stream.LT(1).getType())

        # CALL 104
        stream.push(indexOf104)
        self.failUnlessEqual(104, stream.LT(1).getType())
        stream.consume() # consume 102
        self.failUnlessEqual(DOWN, stream.LT(1).getType())
        stream.consume() # consume DN
        self.failUnlessEqual(105, stream.LT(1).getType())
        stream.consume() # consume 103
        self.failUnlessEqual(UP, stream.LT(1).getType())
        # RETURN (to empty stack)
        stream.pop()
        self.failUnlessEqual(EOF, stream.LT(1).getType())


class TestCommonTree(unittest.TestCase):
    """Test case for the CommonTree class."""

    def setUp(self):
        """Setup test fixure"""

        self.adaptor = CommonTreeAdaptor()

        
    def testSingleNode(self):
        t = CommonTree(CommonToken(101))
        self.failUnless(t.parent is None)
        self.failUnlessEqual(-1, t.childIndex)


    def test4Nodes(self):
        # ^(101 ^(102 103) 104)
        r0 = CommonTree(CommonToken(101))
        r0.addChild(CommonTree(CommonToken(102)))
        r0.getChild(0).addChild(CommonTree(CommonToken(103)))
        r0.addChild(CommonTree(CommonToken(104)))

        self.failUnless(r0.parent is None)
        self.failUnlessEqual(-1, r0.childIndex)


    def testList(self):
        # ^(nil 101 102 103)
        r0 = CommonTree(None)
        c0=CommonTree(CommonToken(101))
        r0.addChild(c0)
        c1=CommonTree(CommonToken(102))
        r0.addChild(c1)
        c2=CommonTree(CommonToken(103))
        r0.addChild(c2)

        self.failUnless(r0.parent is None)
        self.failUnlessEqual(-1, r0.childIndex)
        self.failUnlessEqual(r0, c0.parent)
        self.failUnlessEqual(0, c0.childIndex)
        self.failUnlessEqual(r0, c1.parent)
        self.failUnlessEqual(1, c1.childIndex)        
        self.failUnlessEqual(r0, c2.parent)
        self.failUnlessEqual(2, c2.childIndex)


    def testList2(self):
        # Add child ^(nil 101 102 103) to root 5
        # should pull 101 102 103 directly to become 5's child list
        root = CommonTree(CommonToken(5))

        # child tree
        r0 = CommonTree(None)
        c0=CommonTree(CommonToken(101))
        r0.addChild(c0)
        c1=CommonTree(CommonToken(102))
        r0.addChild(c1)
        c2=CommonTree(CommonToken(103))
        r0.addChild(c2)

        root.addChild(r0)

        self.failUnless(root.parent is None)
        self.failUnlessEqual(-1, root.childIndex)
        # check children of root all point at root
        self.failUnlessEqual(root, c0.parent)
        self.failUnlessEqual(0, c0.childIndex)
        self.failUnlessEqual(root, c0.parent)
        self.failUnlessEqual(1, c1.childIndex)
        self.failUnlessEqual(root, c0.parent)
        self.failUnlessEqual(2, c2.childIndex)


    def testAddListToExistChildren(self):
        # Add child ^(nil 101 102 103) to root ^(5 6)
        # should add 101 102 103 to end of 5's child list
        root = CommonTree(CommonToken(5))
        root.addChild(CommonTree(CommonToken(6)))

        # child tree
        r0 = CommonTree(None)
        c0=CommonTree(CommonToken(101))
        r0.addChild(c0)
        c1=CommonTree(CommonToken(102))
        r0.addChild(c1)
        c2=CommonTree(CommonToken(103))
        r0.addChild(c2)

        root.addChild(r0)

        self.failUnless(root.parent is None)
        self.failUnlessEqual(-1, root.childIndex)
        # check children of root all point at root
        self.failUnlessEqual(root, c0.parent)
        self.failUnlessEqual(1, c0.childIndex)
        self.failUnlessEqual(root, c0.parent)
        self.failUnlessEqual(2, c1.childIndex)
        self.failUnlessEqual(root, c0.parent)
        self.failUnlessEqual(3, c2.childIndex)


    def testDupTree(self):
        # ^(101 ^(102 103 ^(106 107) ) 104 105)
        r0 = CommonTree(CommonToken(101))
        r1 = CommonTree(CommonToken(102))
        r0.addChild(r1)
        r1.addChild(CommonTree(CommonToken(103)))
        r2 = CommonTree(CommonToken(106))
        r2.addChild(CommonTree(CommonToken(107)))
        r1.addChild(r2)
        r0.addChild(CommonTree(CommonToken(104)))
        r0.addChild(CommonTree(CommonToken(105)))

        dup = self.adaptor.dupTree(r0)

        self.failUnless(dup.parent is None)
        self.failUnlessEqual(-1, dup.childIndex)
        dup.sanityCheckParentAndChildIndexes()


    def testBecomeRoot(self):
        # 5 becomes root of ^(nil 101 102 103)
        newRoot = CommonTree(CommonToken(5))

        oldRoot = CommonTree(None)
        oldRoot.addChild(CommonTree(CommonToken(101)))
        oldRoot.addChild(CommonTree(CommonToken(102)))
        oldRoot.addChild(CommonTree(CommonToken(103)))

        self.adaptor.becomeRoot(newRoot, oldRoot)
        newRoot.sanityCheckParentAndChildIndexes()


    def testBecomeRoot2(self):
        # 5 becomes root of ^(101 102 103)
        newRoot = CommonTree(CommonToken(5))

        oldRoot = CommonTree(CommonToken(101))
        oldRoot.addChild(CommonTree(CommonToken(102)))
        oldRoot.addChild(CommonTree(CommonToken(103)))

        self.adaptor.becomeRoot(newRoot, oldRoot)
        newRoot.sanityCheckParentAndChildIndexes()


    def testBecomeRoot3(self):
        # ^(nil 5) becomes root of ^(nil 101 102 103)
        newRoot = CommonTree(None)
        newRoot.addChild(CommonTree(CommonToken(5)))

        oldRoot = CommonTree(None)
        oldRoot.addChild(CommonTree(CommonToken(101)))
        oldRoot.addChild(CommonTree(CommonToken(102)))
        oldRoot.addChild(CommonTree(CommonToken(103)))

        self.adaptor.becomeRoot(newRoot, oldRoot)
        newRoot.sanityCheckParentAndChildIndexes()


    def testBecomeRoot5(self):
        # ^(nil 5) becomes root of ^(101 102 103)
        newRoot = CommonTree(None)
        newRoot.addChild(CommonTree(CommonToken(5)))

        oldRoot = CommonTree(CommonToken(101))
        oldRoot.addChild(CommonTree(CommonToken(102)))
        oldRoot.addChild(CommonTree(CommonToken(103)))

        self.adaptor.becomeRoot(newRoot, oldRoot)
        newRoot.sanityCheckParentAndChildIndexes()


    def testBecomeRoot6(self):
        # emulates construction of ^(5 6)
        root_0 = self.adaptor.nil()
        root_1 = self.adaptor.nil()
        root_1 = self.adaptor.becomeRoot(CommonTree(CommonToken(5)), root_1)

        self.adaptor.addChild(root_1, CommonTree(CommonToken(6)))

        self.adaptor.addChild(root_0, root_1)

        root_0.sanityCheckParentAndChildIndexes()


    # Test replaceChildren

    def testReplaceWithNoChildren(self):
        t = CommonTree(CommonToken(101))
        newChild = CommonTree(CommonToken(5))
        error = False
        try:
        	t.replaceChildren(0, 0, newChild)
	
        except IndexError:
        	error = True
	
        self.failUnless(error)


    def testReplaceWithOneChildren(self):
        # assume token type 99 and use text
        t = CommonTree(CommonToken(99, text="a"))
        c0 = CommonTree(CommonToken(99, text="b"))
        t.addChild(c0)

        newChild = CommonTree(CommonToken(99, text="c"))
        t.replaceChildren(0, 0, newChild)
        expecting = "(a c)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceInMiddle(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c"))) # index 1
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChild = CommonTree(CommonToken(99, text="x"))
        t.replaceChildren(1, 1, newChild)
        expecting = "(a b x d)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceAtLeft(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b"))) # index 0
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChild = CommonTree(CommonToken(99, text="x"))
        t.replaceChildren(0, 0, newChild)
        expecting = "(a x c d)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceAtRight(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d"))) # index 2

        newChild = CommonTree(CommonToken(99, text="x"))
        t.replaceChildren(2, 2, newChild)
        expecting = "(a b c x)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceOneWithTwoAtLeft(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChildren = self.adaptor.nil()
        newChildren.addChild(CommonTree(CommonToken(99, text="x")))
        newChildren.addChild(CommonTree(CommonToken(99, text="y")))

        t.replaceChildren(0, 0, newChildren)
        expecting = "(a x y c d)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceOneWithTwoAtRight(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChildren = self.adaptor.nil()
        newChildren.addChild(CommonTree(CommonToken(99, text="x")))
        newChildren.addChild(CommonTree(CommonToken(99, text="y")))

        t.replaceChildren(2, 2, newChildren)
        expecting = "(a b c x y)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceOneWithTwoInMiddle(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChildren = self.adaptor.nil()
        newChildren.addChild(CommonTree(CommonToken(99, text="x")))
        newChildren.addChild(CommonTree(CommonToken(99, text="y")))

        t.replaceChildren(1, 1, newChildren)
        expecting = "(a b x y d)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceTwoWithOneAtLeft(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChild = CommonTree(CommonToken(99, text="x"))

        t.replaceChildren(0, 1, newChild)
        expecting = "(a x d)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceTwoWithOneAtRight(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChild = CommonTree(CommonToken(99, text="x"))

        t.replaceChildren(1, 2, newChild)
        expecting = "(a b x)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceAllWithOne(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChild = CommonTree(CommonToken(99, text="x"))

        t.replaceChildren(0, 2, newChild)
        expecting = "(a x)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


    def testReplaceAllWithTwo(self):
        t = CommonTree(CommonToken(99, text="a"))
        t.addChild(CommonTree(CommonToken(99, text="b")))
        t.addChild(CommonTree(CommonToken(99, text="c")))
        t.addChild(CommonTree(CommonToken(99, text="d")))

        newChildren = self.adaptor.nil()
        newChildren.addChild(CommonTree(CommonToken(99, text="x")))
        newChildren.addChild(CommonTree(CommonToken(99, text="y")))

        t.replaceChildren(0, 2, newChildren)
        expecting = "(a x y)"
        self.failUnlessEqual(expecting, t.toStringTree())
        t.sanityCheckParentAndChildIndexes()


class TestTreeContext(unittest.TestCase):
    """Test the TreeParser.inContext() method"""

    tokenNames = [
        "<invalid>", "<EOR>", "<DOWN>", "<UP>", "VEC", "ASSIGN", "PRINT",
        "PLUS", "MULT", "DOT", "ID", "INT", "WS", "'['", "','", "']'"
        ]

    def testSimpleParent(self):
        tree = "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID[x] (VEC INT[1] INT[2] INT[3]))))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID (VEC INT %x:INT INT))))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "VEC")
        self.assertEquals(expecting, found)


    def testNoParent(self):
        tree = "(PRINT (MULT ID[x] (VEC INT[1] INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(%x:PRINT (MULT ID (VEC INT INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = False
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "VEC")
        self.assertEquals(expecting, found)


    def testParentWithWildcard(self):
        tree = "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID[x] (VEC INT[1] INT[2] INT[3]))))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID (VEC INT %x:INT INT))))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "VEC ...")
        self.assertEquals(expecting, found)


    def testWildcardAtStartIgnored(self):
        tree = "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID[x] (VEC INT[1] INT[2] INT[3]))))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID (VEC INT %x:INT INT))))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "...VEC")
        self.assertEquals(expecting, found)


    def testWildcardInBetween(self):
        tree = "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID[x] (VEC INT[1] INT[2] INT[3]))))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID (VEC INT %x:INT INT))))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT...VEC")
        self.assertEquals(expecting, found)


    def testLotsOfWildcards(self):
        tree = "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID[x] (VEC INT[1] INT[2] INT[3]))))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(nil (ASSIGN ID[x] INT[3]) (PRINT (MULT ID (VEC INT %x:INT INT))))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "... PRINT ... VEC ...")
        self.assertEquals(expecting, found)


    def testDeep(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "VEC ...")
        self.assertEquals(expecting, found)


    def testDeepAndFindRoot(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT ...")
        self.assertEquals(expecting, found)


    def testDeepAndFindRoot2(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT ... VEC ...")
        self.assertEquals(expecting, found)


    def testChain(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = True
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT MULT VEC MULT")
        self.assertEquals(expecting, found)


    ## TEST INVALID CONTEXTS

    def testNotParent(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = False
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "VEC")
        self.assertEquals(expecting, found)


    def testMismatch(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = False
        ## missing MULT
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT VEC MULT")
        self.assertEquals(expecting, found)


    def testMismatch2(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = False
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT VEC ...")
        self.assertEquals(expecting, found)


    def testMismatch3(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        expecting = False
        found = TreeParser._inContext(adaptor, self.tokenNames, node, "VEC ... VEC MULT")
        self.assertEquals(expecting, found)


    def testDoubleEtc(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        try:
            TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT ... ... VEC")
            self.fail()
        except ValueError, exc: 
            expecting = "invalid syntax: ... ..."
            found = str(exc)
            self.assertEquals(expecting, found)


    def testDotDot(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        labels = {}
        valid = wiz.parse(
            t,
            "(PRINT (MULT ID (VEC (MULT INT %x:INT) INT INT)))",
            labels)
        self.assertTrue(valid)
        node = labels.get("x")

        try:
            TreeParser._inContext(adaptor, self.tokenNames, node, "PRINT .. VEC")
            self.fail()
        except ValueError, exc:
            expecting = "invalid syntax: .."
            found = str(exc)
            self.assertEquals(expecting, found)


class TestTreeVisitor(unittest.TestCase):
    """Test of the TreeVisitor class."""

    tokenNames = [
        "<invalid>", "<EOR>", "<DOWN>", "<UP>", "VEC", "ASSIGN", "PRINT",
        "PLUS", "MULT", "DOT", "ID", "INT", "WS", "'['", "','", "']'"
        ]

    def testTreeVisitor(self):
        tree = "(PRINT (MULT ID[x] (VEC (MULT INT[9] INT[1]) INT[2] INT[3])))"
        adaptor = CommonTreeAdaptor()
        wiz = TreeWizard(adaptor, self.tokenNames)
        t = wiz.create(tree)

        found = []
        def pre(t):
            found.append("pre(%s)" % t)
            return t
        def post(t):
            found.append("post(%s)" % t)
            return t

        visitor = TreeVisitor(adaptor)
        visitor.visit(t, pre, post)

        expecting = [ "pre(PRINT)", "pre(MULT)", "pre(x)", "post(x)",
                      "pre(VEC)", "pre(MULT)", "pre(9)", "post(9)", "pre(1)",
                      "post(1)", "post(MULT)", "pre(2)", "post(2)", "pre(3)",
                      "post(3)", "post(VEC)", "post(MULT)", "post(PRINT)" ]

        self.assertEquals(expecting, found)

if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
