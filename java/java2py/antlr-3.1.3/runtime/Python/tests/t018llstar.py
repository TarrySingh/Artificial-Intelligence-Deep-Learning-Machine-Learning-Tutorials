import antlr3
import testbase
import unittest
import os
import sys
from cStringIO import StringIO
import difflib

class t018llstar(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()
        

    def testValid(self):
        inputPath = os.path.splitext(__file__)[0] + '.input'
        cStream = antlr3.StringStream(open(inputPath).read())
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        parser.program()

        output = parser.output.getvalue()

        outputPath = os.path.splitext(__file__)[0] + '.output'
        testOutput = open(outputPath).read()

        success = (output == testOutput)
        if not success:
            d = difflib.Differ()
            r = d.compare(output.splitlines(1), testOutput.splitlines(1))
            self.fail(
                ''.join([l.encode('ascii', 'backslashreplace') for l in r])
                )

if __name__ == '__main__':
    unittest.main()


    
## # run an infinite loop with randomly mangled input
## while True:
##     print "ping"

##     input = open(inputPath).read()

##     import random
##     input = list(input) # make it mutable
##     for _ in range(3):
##         p1 = random.randrange(len(input))
##         p2 = random.randrange(len(input))

##         c1 = input[p1]
##         input[p1] = input[p2]
##         input[p2] = c1
##     input = ''.join(input) # back to string

        
##     try:
##         cStream = antlr3.StringStream(input)
##         lexer = Lexer(cStream)
##         tStream = antlr3.CommonTokenStream(lexer)
##         parser = TestParser(tStream)
##         parser.program()

##     except antlr3.RecognitionException, exc:
##         print exc
##         for l in input.splitlines()[0:exc.line]:
##             print l
##         print ' '*exc.charPositionInLine + '^'

##     except BaseException, exc:
##         print '\n'.join(['%02d: %s' % (idx+1, l) for idx, l in enumerate(input.splitlines())])
##         print "%s at %d:%d" % (exc, cStream.line, cStream.charPositionInLine)
##         print
        
##         raise
