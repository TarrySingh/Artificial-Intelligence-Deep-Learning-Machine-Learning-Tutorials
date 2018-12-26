"""Testsuite for TokenRewriteStream class."""

# don't care about docstrings
# pylint: disable-msg=C0111

import unittest
import antlr3
import testbase

class T1(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()


    def _parse(self, input):
        cStream = antlr3.StringStream(input)
        lexer = self.getLexer(cStream)
        tStream = antlr3.TokenRewriteStream(lexer)
        tStream.LT(1) # fill buffer

        return tStream

   
    def testInsertBeforeIndex0(self):
        tokens = self._parse("abc")
        tokens.insertBefore(0, "0")
        
        result = tokens.toString()
        expecting = "0abc"
        self.failUnlessEqual(result, expecting)


    def testInsertAfterLastIndex(self):
        tokens = self._parse("abc")
        tokens.insertAfter(2, "x")
        
        result = tokens.toString()
        expecting = "abcx"
        self.failUnlessEqual(result, expecting)


    def test2InsertBeforeAfterMiddleIndex(self):
        tokens = self._parse("abc")
        tokens.insertBefore(1, "x")
        tokens.insertAfter(1, "x")
        
        result = tokens.toString()
        expecting = "axbxc"
        self.failUnlessEqual(result, expecting)


    def testReplaceIndex0(self):
        tokens = self._parse("abc")
        tokens.replace(0, "x")

        result = tokens.toString()
        expecting = "xbc"
        self.failUnlessEqual(result, expecting)


    def testReplaceLastIndex(self):
        tokens = self._parse("abc")
        tokens.replace(2, "x")

        result = tokens.toString()
        expecting = "abx"
        self.failUnlessEqual(result, expecting)


    def testReplaceMiddleIndex(self):
        tokens = self._parse("abc")
        tokens.replace(1, "x")

        result = tokens.toString()
        expecting = "axc"
        self.failUnlessEqual(result, expecting)


    def test2ReplaceMiddleIndex(self):
        tokens = self._parse("abc")
        tokens.replace(1, "x")
        tokens.replace(1, "y")

        result = tokens.toString()
        expecting = "ayc"
        self.failUnlessEqual(result, expecting)


    def test2ReplaceMiddleIndex1InsertBefore(self):
        tokens = self._parse("abc")
        tokens.insertBefore(0, "_")
        tokens.replace(1, "x")
        tokens.replace(1, "y")
        
        result = tokens.toString()
        expecting = "_ayc"
        self.failUnlessEqual(expecting, result)


    def testReplaceThenDeleteMiddleIndex(self):
        tokens = self._parse("abc")
        tokens.replace(1, "x")
        tokens.delete(1)

        result = tokens.toString()
        expecting = "ac"
        self.failUnlessEqual(result, expecting)


    def testInsertInPriorReplace(self):
        tokens = self._parse("abc")
        tokens.replace(0, 2, "x")
        tokens.insertBefore(1, "0")
        try:
            tokens.toString()
            self.fail()
        except ValueError, exc:
            self.failUnlessEqual(
                str(exc),
                "insert op <InsertBeforeOp@1:\"0\"> within boundaries of "
                "previous <ReplaceOp@0..2:\"x\">"
                )

    def testInsertThenReplaceSameIndex(self):
        tokens = self._parse("abc")
        tokens.insertBefore(0, "0")
        tokens.replace(0, "x")  # supercedes insert at 0

        result = tokens.toString()
        expecting = "xbc"
        self.failUnlessEqual(result, expecting)


    def test2InsertMiddleIndex(self):
        tokens = self._parse("abc")
        tokens.insertBefore(1, "x")
        tokens.insertBefore(1, "y")

        result = tokens.toString()
        expecting = "ayxbc"
        self.failUnlessEqual(result, expecting)


    def test2InsertThenReplaceIndex0(self):
        tokens = self._parse("abc")
        tokens.insertBefore(0, "x")
        tokens.insertBefore(0, "y")
        tokens.replace(0, "z")

        result = tokens.toString()
        expecting = "zbc"
        self.failUnlessEqual(result, expecting)


    def testReplaceThenInsertBeforeLastIndex(self):
        tokens = self._parse("abc")
        tokens.replace(2, "x")
        tokens.insertBefore(2, "y")

        result = tokens.toString()
        expecting = "abyx"
        self.failUnlessEqual(result, expecting)


    def testInsertThenReplaceLastIndex(self):
        tokens = self._parse("abc")
        tokens.insertBefore(2, "y")
        tokens.replace(2, "x")

        result = tokens.toString()
        expecting = "abx"
        self.failUnlessEqual(result, expecting)


    def testReplaceThenInsertAfterLastIndex(self):
        tokens = self._parse("abc")
        tokens.replace(2, "x")
        tokens.insertAfter(2, "y")
        
        result = tokens.toString()
        expecting = "abxy"
        self.failUnlessEqual(result, expecting)


    def testReplaceRangeThenInsertAtLeftEdge(self):
        tokens = self._parse("abcccba")
        tokens.replace(2, 4, "x")
        tokens.insertBefore(2, "y")

        result = tokens.toString()
        expecting = "abyxba"
        self.failUnlessEqual(result, expecting)


    def testReplaceRangeThenInsertAtRightEdge(self):
        tokens = self._parse("abcccba")
        tokens.replace(2, 4, "x")
        tokens.insertBefore(4, "y") # no effect; within range of a replace

        try:
            tokens.toString()
            self.fail()
        except ValueError, exc:
            self.failUnlessEqual(
                str(exc),
                "insert op <InsertBeforeOp@4:\"y\"> within boundaries of "
                "previous <ReplaceOp@2..4:\"x\">")


    def testReplaceRangeThenInsertAfterRightEdge(self):
        tokens = self._parse("abcccba")
        tokens.replace(2, 4, "x")
        tokens.insertAfter(4, "y")

        result = tokens.toString()
        expecting = "abxyba"
        self.failUnlessEqual(result, expecting)


    def testReplaceAll(self):
        tokens = self._parse("abcccba")
        tokens.replace(0, 6, "x")

        result = tokens.toString()
        expecting = "x"
        self.failUnlessEqual(result, expecting)


    def testReplaceSubsetThenFetch(self):
        tokens = self._parse("abcccba")
        tokens.replace(2, 4, "xyz")

        result = tokens.toString(0, 6)
        expecting = "abxyzba"
        self.failUnlessEqual(result, expecting)


    def testReplaceThenReplaceSuperset(self):
        tokens = self._parse("abcccba")
        tokens.replace(2, 4, "xyz")
        tokens.replace(3, 5, "foo") # overlaps, error

        try:
            tokens.toString()
            self.fail()
        except ValueError, exc:
            self.failUnlessEqual(
                str(exc),
                "replace op boundaries of <ReplaceOp@3..5:\"foo\"> overlap "
                "with previous <ReplaceOp@2..4:\"xyz\">")


    def testReplaceThenReplaceLowerIndexedSuperset(self):
        tokens = self._parse("abcccba")
        tokens.replace(2, 4, "xyz")
        tokens.replace(1, 3, "foo") # overlap, error

        try:
            tokens.toString()
            self.fail()
        except ValueError, exc:
            self.failUnlessEqual(
                str(exc),
                "replace op boundaries of <ReplaceOp@1..3:\"foo\"> overlap "
                "with previous <ReplaceOp@2..4:\"xyz\">")


    def testReplaceSingleMiddleThenOverlappingSuperset(self):
        tokens = self._parse("abcba")
        tokens.replace(2, 2, "xyz")
        tokens.replace(0, 3, "foo")
        
        result = tokens.toString()
        expecting = "fooa"
        self.failUnlessEqual(result, expecting)


    def testCombineInserts(self):
        tokens = self._parse("abc")
        tokens.insertBefore(0, "x")
        tokens.insertBefore(0, "y")
        result = tokens.toString()
        expecting = "yxabc"
        self.failUnlessEqual(expecting, result)


    def testCombine3Inserts(self):
        tokens = self._parse("abc")
        tokens.insertBefore(1, "x")
        tokens.insertBefore(0, "y")
        tokens.insertBefore(1, "z")
        result = tokens.toString()
        expecting = "yazxbc"
        self.failUnlessEqual(expecting, result)


    def testCombineInsertOnLeftWithReplace(self):
        tokens = self._parse("abc")
        tokens.replace(0, 2, "foo")
        tokens.insertBefore(0, "z") # combine with left edge of rewrite
        result = tokens.toString()
        expecting = "zfoo"
        self.failUnlessEqual(expecting, result)


    def testCombineInsertOnLeftWithDelete(self):
        tokens = self._parse("abc")
        tokens.delete(0, 2)
        tokens.insertBefore(0, "z") # combine with left edge of rewrite
        result = tokens.toString()
        expecting = "z" # make sure combo is not znull
        self.failUnlessEqual(expecting, result)


    def testDisjointInserts(self):
        tokens = self._parse("abc")
        tokens.insertBefore(1, "x")
        tokens.insertBefore(2, "y")
        tokens.insertBefore(0, "z")
        result = tokens.toString()
        expecting = "zaxbyc"
        self.failUnlessEqual(expecting, result)


    def testOverlappingReplace(self):
        tokens = self._parse("abcc")
        tokens.replace(1, 2, "foo")
        tokens.replace(0, 3, "bar") # wipes prior nested replace
        result = tokens.toString()
        expecting = "bar"
        self.failUnlessEqual(expecting, result)


    def testOverlappingReplace2(self):
        tokens = self._parse("abcc")
        tokens.replace(0, 3, "bar")
        tokens.replace(1, 2, "foo") # cannot split earlier replace

        try:
            tokens.toString()
            self.fail()
        except ValueError, exc:
            self.failUnlessEqual(
                str(exc),
                "replace op boundaries of <ReplaceOp@1..2:\"foo\"> overlap "
                "with previous <ReplaceOp@0..3:\"bar\">")


    def testOverlappingReplace3(self):
        tokens = self._parse("abcc")
        tokens.replace(1, 2, "foo")
        tokens.replace(0, 2, "bar") # wipes prior nested replace
        result = tokens.toString()
        expecting = "barc"
        self.failUnlessEqual(expecting, result)


    def testOverlappingReplace4(self):
        tokens = self._parse("abcc")
        tokens.replace(1, 2, "foo")
        tokens.replace(1, 3, "bar") # wipes prior nested replace
        result = tokens.toString()
        expecting = "abar"
        self.failUnlessEqual(expecting, result)


    def testDropIdenticalReplace(self):
        tokens = self._parse("abcc")
        tokens.replace(1, 2, "foo")
        tokens.replace(1, 2, "foo") # drop previous, identical
        result = tokens.toString()
        expecting = "afooc"
        self.failUnlessEqual(expecting, result)


    def testDropPrevCoveredInsert(self):
        tokens = self._parse("abcc")
        tokens.insertBefore(1, "foo")
        tokens.replace(1, 2, "foo") # kill prev insert
        result = tokens.toString()
        expecting = "afooc"
        self.failUnlessEqual(expecting, result)


    def testLeaveAloneDisjointInsert(self):
        tokens = self._parse("abcc")
        tokens.insertBefore(1, "x")
        tokens.replace(2, 3, "foo")
        result = tokens.toString()
        expecting = "axbfoo"
        self.failUnlessEqual(expecting, result)


    def testLeaveAloneDisjointInsert2(self):
        tokens = self._parse("abcc")
        tokens.replace(2, 3, "foo")
        tokens.insertBefore(1, "x")
        result = tokens.toString()
        expecting = "axbfoo"
        self.failUnlessEqual(expecting, result)


class T2(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar('t048rewrite2.g')


    def _parse(self, input):
        cStream = antlr3.StringStream(input)
        lexer = self.getLexer(cStream)
        tStream = antlr3.TokenRewriteStream(lexer)
        tStream.LT(1) # fill buffer

        return tStream

   
    def testToStringStartStop(self):
        # Tokens: 0123456789
        # Input:  x = 3 * 0
        tokens = self._parse("x = 3 * 0;")
        tokens.replace(4, 8, "0") # replace 3 * 0 with 0

        result = tokens.toOriginalString()
        expecting = "x = 3 * 0;"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString()
        expecting = "x = 0;"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(0, 9)
        expecting = "x = 0;"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(4, 8)
        expecting = "0"
        self.failUnlessEqual(expecting, result)


    def testToStringStartStop2(self):
        # Tokens: 012345678901234567
        # Input:  x = 3 * 0 + 2 * 0
        tokens = self._parse("x = 3 * 0 + 2 * 0;")

        result = tokens.toOriginalString()
        expecting = "x = 3 * 0 + 2 * 0;"
        self.failUnlessEqual(expecting, result)

        tokens.replace(4, 8, "0") # replace 3 * 0 with 0
        result = tokens.toString()
        expecting = "x = 0 + 2 * 0;"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(0, 17)
        expecting = "x = 0 + 2 * 0;"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(4, 8)
        expecting = "0"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(0, 8)
        expecting = "x = 0"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(12, 16)
        expecting = "2 * 0"
        self.failUnlessEqual(expecting, result)

        tokens.insertAfter(17, "// comment")
        result = tokens.toString(12, 17)
        expecting = "2 * 0;// comment"
        self.failUnlessEqual(expecting, result)

        result = tokens.toString(0, 8) # try again after insert at end
        expecting = "x = 0"
        self.failUnlessEqual(expecting, result)


if __name__ == '__main__':
    unittest.main()

