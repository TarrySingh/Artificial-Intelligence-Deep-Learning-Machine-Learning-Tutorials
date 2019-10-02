import antlr3
import testbase
import unittest

class t015calc(testbase.ANTLRTest):
    def setUp(self):
        self.compileGrammar()


    def _evaluate(self, expr, expected, errors=[]):
        cStream = antlr3.StringStream(expr)
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        result = parser.evaluate()
        assert result == expected, "%r != %r" % (result, expected)
        assert len(parser.reportedErrors) == len(errors), parser.reportedErrors


    def testValid01(self):
        self._evaluate("1 + 2", 3)


    def testValid02(self):
        self._evaluate("1 + 2 * 3", 7)


    def testValid03(self):
        self._evaluate("10 / 2", 5)


    def testValid04(self):
        self._evaluate("6 + 2*(3+1) - 4", 10)


    def testMalformedInput(self):
        self._evaluate("6 - (2*1", 4, ["mismatched token at pos 8"])
        
    # FIXME: most parse errors result in TypeErrors in action code, because
    # rules return None, which is then added/multiplied... to integers.
    # evaluate("6 - foo 2", 4, ["some error"])
            

if __name__ == '__main__':
    unittest.main()

