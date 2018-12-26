# -*- coding: utf-8 -*-

import os
import unittest
from StringIO import StringIO
import antlr3


class TestStringStream(unittest.TestCase):
    """Test case for the StringStream class."""

    def testSize(self):
        """StringStream.size()"""

        stream = antlr3.StringStream('foo')

        self.failUnlessEqual(stream.size(), 3)

        
    def testIndex(self):
        """StringStream.index()"""

        stream = antlr3.StringStream('foo')

        self.failUnlessEqual(stream.index(), 0)
        
        
    def testConsume(self):
        """StringStream.consume()"""

        stream = antlr3.StringStream('foo\nbar')

        stream.consume() # f
        self.failUnlessEqual(stream.index(), 1)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.line, 1)

        stream.consume() # o
        self.failUnlessEqual(stream.index(), 2)
        self.failUnlessEqual(stream.charPositionInLine, 2)
        self.failUnlessEqual(stream.line, 1)

        stream.consume() # o
        self.failUnlessEqual(stream.index(), 3)
        self.failUnlessEqual(stream.charPositionInLine, 3)
        self.failUnlessEqual(stream.line, 1)

        stream.consume() # \n
        self.failUnlessEqual(stream.index(), 4)
        self.failUnlessEqual(stream.charPositionInLine, 0)
        self.failUnlessEqual(stream.line, 2)

        stream.consume() # b
        self.failUnlessEqual(stream.index(), 5)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.line, 2)

        stream.consume() # a
        self.failUnlessEqual(stream.index(), 6)
        self.failUnlessEqual(stream.charPositionInLine, 2)
        self.failUnlessEqual(stream.line, 2)

        stream.consume() # r
        self.failUnlessEqual(stream.index(), 7)
        self.failUnlessEqual(stream.charPositionInLine, 3)
        self.failUnlessEqual(stream.line, 2)

        stream.consume() # EOF
        self.failUnlessEqual(stream.index(), 7)
        self.failUnlessEqual(stream.charPositionInLine, 3)
        self.failUnlessEqual(stream.line, 2)

        stream.consume() # EOF
        self.failUnlessEqual(stream.index(), 7)
        self.failUnlessEqual(stream.charPositionInLine, 3)
        self.failUnlessEqual(stream.line, 2)
        
        
    def testReset(self):
        """StringStream.reset()"""

        stream = antlr3.StringStream('foo')

        stream.consume()
        stream.consume()

        stream.reset()
        self.failUnlessEqual(stream.index(), 0)
        self.failUnlessEqual(stream.line, 1)
        self.failUnlessEqual(stream.charPositionInLine, 0)
        self.failUnlessEqual(stream.LT(1), 'f')
        
        
    def testLA(self):
        """StringStream.LA()"""

        stream = antlr3.StringStream('foo')

        self.failUnlessEqual(stream.LT(1), 'f')
        self.failUnlessEqual(stream.LT(2), 'o')
        self.failUnlessEqual(stream.LT(3), 'o')
        
        stream.consume()
        stream.consume()

        self.failUnlessEqual(stream.LT(1), 'o')
        self.failUnlessEqual(stream.LT(2), antlr3.EOF)
        self.failUnlessEqual(stream.LT(3), antlr3.EOF)


    def testSubstring(self):
        """StringStream.substring()"""

        stream = antlr3.StringStream('foobar')
        
        self.failUnlessEqual(stream.substring(0, 0), 'f')
        self.failUnlessEqual(stream.substring(0, 1), 'fo')
        self.failUnlessEqual(stream.substring(0, 5), 'foobar')
        self.failUnlessEqual(stream.substring(3, 5), 'bar')
        
        
    def testSeekForward(self):
        """StringStream.seek(): forward"""

        stream = antlr3.StringStream('foo\nbar')

        stream.seek(4)
        
        self.failUnlessEqual(stream.index(), 4)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 0)
        self.failUnlessEqual(stream.LT(1), 'b')


##     # not yet implemented
##     def testSeekBackward(self):
##         """StringStream.seek(): backward"""

##         stream = antlr3.StringStream('foo\nbar')

##         stream.seek(4)
##         stream.seek(1)
        
##         self.failUnlessEqual(stream.index(), 1)
##         self.failUnlessEqual(stream.line, 1)
##         self.failUnlessEqual(stream.charPositionInLine, 1)
##         self.failUnlessEqual(stream.LA(1), 'o')


    def testMark(self):
        """StringStream.mark()"""

        stream = antlr3.StringStream('foo\nbar')

        stream.seek(4)

        marker = stream.mark()
        self.failUnlessEqual(marker, 1)
        self.failUnlessEqual(stream.markDepth, 1)

        stream.consume()
        marker = stream.mark()
        self.failUnlessEqual(marker, 2)
        self.failUnlessEqual(stream.markDepth, 2)
        

    def testReleaseLast(self):
        """StringStream.release(): last marker"""

        stream = antlr3.StringStream('foo\nbar')

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()

        stream.release()
        self.failUnlessEqual(stream.markDepth, 1)

        # release same marker again, nothing has changed
        stream.release()
        self.failUnlessEqual(stream.markDepth, 1)
        

    def testReleaseNested(self):
        """StringStream.release(): nested"""

        stream = antlr3.StringStream('foo\nbar')

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()
        
        stream.consume()
        marker3 = stream.mark()

        stream.release(marker2)
        self.failUnlessEqual(stream.markDepth, 1)
        

    def testRewindLast(self):
        """StringStream.rewind(): last marker"""

        stream = antlr3.StringStream('foo\nbar')

        stream.seek(4)

        marker = stream.mark()
        stream.consume()
        stream.consume()

        stream.rewind()
        self.failUnlessEqual(stream.markDepth, 0)
        self.failUnlessEqual(stream.index(), 4)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 0)
        self.failUnlessEqual(stream.LT(1), 'b')
        

    def testRewindNested(self):
        """StringStream.rewind(): nested"""

        stream = antlr3.StringStream('foo\nbar')

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()
        
        stream.consume()
        marker3 = stream.mark()

        stream.rewind(marker2)
        self.failUnlessEqual(stream.markDepth, 1)
        self.failUnlessEqual(stream.index(), 5)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.LT(1), 'a')

        
class TestFileStream(unittest.TestCase):
    """Test case for the FileStream class."""


    def testNoEncoding(self):
        path = os.path.join(os.path.dirname(__file__), 'teststreams.input1')
        
        stream = antlr3.FileStream(path)

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()
        
        stream.consume()
        marker3 = stream.mark()

        stream.rewind(marker2)
        self.failUnlessEqual(stream.markDepth, 1)
        self.failUnlessEqual(stream.index(), 5)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.LT(1), 'a')
        self.failUnlessEqual(stream.LA(1), ord('a'))


    def testEncoded(self):
        path = os.path.join(os.path.dirname(__file__), 'teststreams.input2')
        
        stream = antlr3.FileStream(path, 'utf-8')

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()
        
        stream.consume()
        marker3 = stream.mark()

        stream.rewind(marker2)
        self.failUnlessEqual(stream.markDepth, 1)
        self.failUnlessEqual(stream.index(), 5)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.LT(1), u'ä')
        self.failUnlessEqual(stream.LA(1), ord(u'ä'))

        

class TestInputStream(unittest.TestCase):
    """Test case for the InputStream class."""

    def testNoEncoding(self):
        file = StringIO('foo\nbar')
        
        stream = antlr3.InputStream(file)

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()
        
        stream.consume()
        marker3 = stream.mark()

        stream.rewind(marker2)
        self.failUnlessEqual(stream.markDepth, 1)
        self.failUnlessEqual(stream.index(), 5)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.LT(1), 'a')
        self.failUnlessEqual(stream.LA(1), ord('a'))


    def testEncoded(self):
        file = StringIO(u'foo\nbär'.encode('utf-8'))
        
        stream = antlr3.InputStream(file, 'utf-8')

        stream.seek(4)
        marker1 = stream.mark()
        
        stream.consume()
        marker2 = stream.mark()
        
        stream.consume()
        marker3 = stream.mark()

        stream.rewind(marker2)
        self.failUnlessEqual(stream.markDepth, 1)
        self.failUnlessEqual(stream.index(), 5)
        self.failUnlessEqual(stream.line, 2)
        self.failUnlessEqual(stream.charPositionInLine, 1)
        self.failUnlessEqual(stream.LT(1), u'ä')
        self.failUnlessEqual(stream.LA(1), ord(u'ä'))

        
class TestCommonTokenStream(unittest.TestCase):
    """Test case for the StringStream class."""

    def setUp(self):
        """Setup test fixure

        The constructor of CommonTokenStream needs a token source. This
        is a simple mock class providing just the nextToken() method.

        """

        class MockSource(object):
            def __init__(self):
                self.tokens = []

            def nextToken(self):
                try:
                    return self.tokens.pop(0)
                except IndexError:
                    return None
                
        self.source = MockSource()
        
        
    def testInit(self):
        """CommonTokenStream.__init__()"""

        stream = antlr3.CommonTokenStream(self.source)
        self.failUnlessEqual(stream.index(), -1)
        
        
    def testSetTokenSource(self):
        """CommonTokenStream.setTokenSource()"""

        stream = antlr3.CommonTokenStream(None)
        stream.setTokenSource(self.source)
        self.failUnlessEqual(stream.index(), -1)
        self.failUnlessEqual(stream.channel, antlr3.DEFAULT_CHANNEL)


    def testLTEmptySource(self):
        """CommonTokenStream.LT(): EOF (empty source)"""

        stream = antlr3.CommonTokenStream(self.source)

        lt1 = stream.LT(1)
        self.failUnlessEqual(lt1.type, antlr3.EOF)
        

    def testLT1(self):
        """CommonTokenStream.LT(1)"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        stream = antlr3.CommonTokenStream(self.source)

        lt1 = stream.LT(1)
        self.failUnlessEqual(lt1.type, 12)
        

    def testLT1WithHidden(self):
        """CommonTokenStream.LT(1): with hidden tokens"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12, channel=antlr3.HIDDEN_CHANNEL)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        stream = antlr3.CommonTokenStream(self.source)

        lt1 = stream.LT(1)
        self.failUnlessEqual(lt1.type, 13)
        

    def testLT2BeyondEnd(self):
        """CommonTokenStream.LT(2): beyond end"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13, channel=antlr3.HIDDEN_CHANNEL)
            )
        
        stream = antlr3.CommonTokenStream(self.source)

        lt1 = stream.LT(2)
        self.failUnlessEqual(lt1.type, antlr3.EOF)
        

    # not yet implemented
    def testLTNegative(self):
        """CommonTokenStream.LT(-1): look back"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        stream.fillBuffer()
        stream.consume()
        
        lt1 = stream.LT(-1)
        self.failUnlessEqual(lt1.type, 12)


    def testLB1(self):
        """CommonTokenStream.LB(1)"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        stream.fillBuffer()
        stream.consume()
        
        self.failUnlessEqual(stream.LB(1).type, 12)
        

    def testLTZero(self):
        """CommonTokenStream.LT(0)"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        stream = antlr3.CommonTokenStream(self.source)

        lt1 = stream.LT(0)
        self.failUnless(lt1 is None)
        

    def testLBBeyondBegin(self):
        """CommonTokenStream.LB(-1): beyond begin"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=12, channel=antlr3.HIDDEN_CHANNEL)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=12, channel=antlr3.HIDDEN_CHANNEL)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        self.failUnless(stream.LB(1) is None)

        stream.consume()
        stream.consume()
        self.failUnless(stream.LB(3) is None)
        

    def testFillBuffer(self):
        """CommonTokenStream.fillBuffer()"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=14)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=antlr3.EOF)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        stream.fillBuffer()

        self.failUnlessEqual(len(stream.tokens), 3)
        self.failUnlessEqual(stream.tokens[0].type, 12)
        self.failUnlessEqual(stream.tokens[1].type, 13)
        self.failUnlessEqual(stream.tokens[2].type, 14)
        
        
    def testConsume(self):
        """CommonTokenStream.consume()"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=antlr3.EOF)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        self.failUnlessEqual(stream.LA(1), 12)

        stream.consume()
        self.failUnlessEqual(stream.LA(1), 13)

        stream.consume()
        self.failUnlessEqual(stream.LA(1), antlr3.EOF)

        stream.consume()
        self.failUnlessEqual(stream.LA(1), antlr3.EOF)
        
        
    def testSeek(self):
        """CommonTokenStream.seek()"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=antlr3.EOF)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        self.failUnlessEqual(stream.LA(1), 12)

        stream.seek(2)
        self.failUnlessEqual(stream.LA(1), antlr3.EOF)

        stream.seek(0)
        self.failUnlessEqual(stream.LA(1), 12)
        
        
    def testMarkRewind(self):
        """CommonTokenStream.mark()/rewind()"""

        self.source.tokens.append(
            antlr3.CommonToken(type=12)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13)
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=antlr3.EOF)
            )
        
        stream = antlr3.CommonTokenStream(self.source)
        stream.fillBuffer()
        
        stream.consume()
        marker = stream.mark()
        
        stream.consume()
        stream.rewind(marker)
        
        self.failUnlessEqual(stream.LA(1), 13)


    def testToString(self):
        """CommonTokenStream.toString()"""
        
        self.source.tokens.append(
            antlr3.CommonToken(type=12, text="foo")
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=13, text="bar")
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=14, text="gnurz")
            )
        
        self.source.tokens.append(
            antlr3.CommonToken(type=15, text="blarz")
            )
        
        stream = antlr3.CommonTokenStream(self.source)

        assert stream.toString() == "foobargnurzblarz"
        assert stream.toString(1, 2) == "bargnurz"
        assert stream.toString(stream.tokens[1], stream.tokens[-2]) == "bargnurz"
        

if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
