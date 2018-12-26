import unittest
import textwrap
import antlr3
import antlr3.tree
import stringtemplate3
import testbase
import sys
import os
from StringIO import StringIO

# FIXME: port other tests from TestLexer.java

class T(testbase.ANTLRTest):
    def execParser(self, grammar, grammarEntry, input):
        lexerCls, parserCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        result = getattr(parser, grammarEntry)()
        return result
    

    def testRefToRuleDoesNotSetChannel(self):
        # this must set channel of A to HIDDEN.  $channel is local to rule
        # like $type.
        grammar = textwrap.dedent(
            r'''
            grammar P;
            options {
              language=Python;
            }
            a returns [foo]: A EOF { $foo = '\%s, channel=\%d' \% ($A.text, $A.channel); } ;
            A : '-' WS I ;
            I : '0'..'9'+ ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(
            grammar, 'a',
            "- 34"
            )

        self.failUnlessEqual("- 34, channel=0", found)

        
if __name__ == '__main__':
    unittest.main()
