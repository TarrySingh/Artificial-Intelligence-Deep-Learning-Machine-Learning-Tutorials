import sys
import unittest

import antlr3


class TestBaseRecognizer(unittest.TestCase):
    """Tests for BaseRecognizer class"""
    
    def testGetRuleInvocationStack(self):
        """BaseRecognizer._getRuleInvocationStack()"""

        rules = antlr3.BaseRecognizer._getRuleInvocationStack(__name__)
        self.failUnlessEqual(
            rules,
            ['testGetRuleInvocationStack']
            )
        

class TestTokenSource(unittest.TestCase):
    """Testcase to the antlr3.TokenSource class"""

    
    def testIteratorInterface(self):
        """TokenSource.next()"""

        class TrivialToken(object):
            def __init__(self, type):
                self.type = type
                
        class TestSource(antlr3.TokenSource):
            def __init__(self):
                self.tokens = [
                    TrivialToken(1),
                    TrivialToken(2),
                    TrivialToken(3),
                    TrivialToken(4),
                    TrivialToken(antlr3.EOF),
                    ]

            def nextToken(self):
                return self.tokens.pop(0)

                
        src = TestSource()
        tokens = []
        for token in src:
            tokens.append(token.type)

        self.failUnlessEqual(tokens, [1, 2, 3, 4])
        
            

class TestLexer(unittest.TestCase):

    def testInit(self):
        """Lexer.__init__()"""

        class TLexer(antlr3.Lexer):
            antlr_version = antlr3.runtime_version

        stream = antlr3.StringStream('foo')
        TLexer(stream)

            
if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
