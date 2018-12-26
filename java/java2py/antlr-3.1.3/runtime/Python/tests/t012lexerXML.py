import antlr3
import testbase
import unittest
import os
import sys
from cStringIO import StringIO
import difflib
import textwrap

class t012lexerXML(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar('t012lexerXMLLexer.g')
        
        
    def lexerClass(self, base):
        class TLexer(base):
            def emitErrorMessage(self, msg):
                # report errors to /dev/null
                pass

            def reportError(self, re):
                # no error recovery yet, just crash!
                raise re

        return TLexer
    
        
    def testValid(self):
        inputPath = os.path.splitext(__file__)[0] + '.input'
        stream = antlr3.StringStream(unicode(open(inputPath).read(), 'utf-8'))
        lexer = self.getLexer(stream)

        while True:
            token = lexer.nextToken()
            if token.type == self.lexerModule.EOF:
                break


        output = unicode(lexer.outbuf.getvalue(), 'utf-8')

        outputPath = os.path.splitext(__file__)[0] + '.output'
        testOutput = unicode(open(outputPath).read(), 'utf-8')

        success = (output == testOutput)
        if not success:
            d = difflib.Differ()
            r = d.compare(output.splitlines(1), testOutput.splitlines(1))
            self.fail(
                ''.join([l.encode('ascii', 'backslashreplace') for l in r])
                )


    def testMalformedInput1(self):
        input = textwrap.dedent("""\
        <?xml version='1.0'?>
        <document d>
        </document>
        """)

        stream = antlr3.StringStream(input)
        lexer = self.getLexer(stream)

        try:
            while True:
                token = lexer.nextToken()
                if token.type == antlr3.EOF:
                    break

            raise AssertionError

        except antlr3.NoViableAltException, exc:
            assert exc.unexpectedType == '>', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 11, repr(exc.charPositionInLine)
            assert exc.line == 2, repr(exc.line)


    def testMalformedInput2(self):
        input = textwrap.dedent("""\
        <?tml version='1.0'?>
        <document>
        </document>
        """)

        stream = antlr3.StringStream(input)
        lexer = self.getLexer(stream)

        try:
            while True:
                token = lexer.nextToken()
                if token.type == antlr3.EOF:
                    break

            raise AssertionError

        except antlr3.MismatchedSetException, exc:
            assert exc.unexpectedType == 't', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 2, repr(exc.charPositionInLine)
            assert exc.line == 1, repr(exc.line)


    def testMalformedInput3(self):
        input = textwrap.dedent("""\
        <?xml version='1.0'?>
        <docu ment attr="foo">
        </document>
        """)

        stream = antlr3.StringStream(input)
        lexer = self.getLexer(stream)

        try:
            while True:
                token = lexer.nextToken()
                if token.type == antlr3.EOF:
                    break

            raise AssertionError

        except antlr3.NoViableAltException, exc:
            assert exc.unexpectedType == 'a', repr(exc.unexpectedType)
            assert exc.charPositionInLine == 11, repr(exc.charPositionInLine)
            assert exc.line == 2, repr(exc.line)

            

if __name__ == '__main__':
    unittest.main()


## # run an infinite loop with randomly mangled input
## while True:
##     print "ping"

##     input = """\
## <?xml version='1.0'?>
## <!DOCTYPE component [
## <!ELEMENT component (PCDATA|sub)*>
## <!ATTLIST component
##           attr CDATA #IMPLIED
##           attr2 CDATA #IMPLIED
## >
## <!ELMENT sub EMPTY>

## ]>
## <component attr="val'ue" attr2='val"ue'>
## <!-- This is a comment -->
## Text
## <![CDATA[huhu]]>
## &amp;
## &lt;
## <?xtal cursor='11'?>
## <sub/>
## <sub></sub>
## </component>
## """

##     import random
##     input = list(input) # make it mutable
##     for _ in range(3):
##         p1 = random.randrange(len(input))
##         p2 = random.randrange(len(input))

##         c1 = input[p1]
##         input[p1] = input[p2]
##         input[p2] = c1
##     input = ''.join(input) # back to string
        
##     stream = antlr3.StringStream(input)
##     lexer = Lexer(stream)

##     try:
##         while True:
##             token = lexer.nextToken()
##             if token.type == EOF:
##                 break

##     except antlr3.RecognitionException, exc:
##         print exc
##         for l in input.splitlines()[0:exc.line]:
##             print l
##         print ' '*exc.charPositionInLine + '^'

##     except BaseException, exc:
##         print '\n'.join(['%02d: %s' % (idx+1, l) for idx, l in enumerate(input.splitlines())])
##         print "%s at %d:%d" % (exc, stream.line, stream.charPositionInLine)
##         print
        
##         raise
    
