import os
import sys
import antlr3
import testbase
import unittest
from cStringIO import StringIO
import difflib

class t020fuzzy(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar('t020fuzzyLexer.g')
        

    def testValid(self):
        inputPath = os.path.splitext(__file__)[0] + '.input'
        stream = antlr3.StringStream(open(inputPath).read())
        lexer = self.getLexer(stream)

        while True:
            token = lexer.nextToken()
            if token.type == antlr3.EOF:
                break


        output = lexer.output.getvalue()

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
