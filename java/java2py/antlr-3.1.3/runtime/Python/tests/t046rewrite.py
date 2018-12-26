import unittest
import textwrap
import antlr3
import testbase

class T(testbase.ANTLRTest):
    def testRewrite(self):
        self.compileGrammar()

        input = textwrap.dedent(
            '''\
            method foo() {
              i = 3;
              k = i;
              i = k*4;
            }

            method bar() {
              j = i*2;
            }
            ''')
        
        cStream = antlr3.StringStream(input)
        lexer = self.getLexer(cStream)
        tStream = antlr3.TokenRewriteStream(lexer)
        parser = self.getParser(tStream)
        parser.program()

        expectedOutput = textwrap.dedent('''\
        public class Wrapper {
        public void foo() {
        int k;
        int i;
          i = 3;
          k = i;
          i = k*4;
        }

        public void bar() {
        int j;
          j = i*2;
        }
        }

        ''')

        self.failUnlessEqual(
            str(tStream),
            expectedOutput
            )


if __name__ == '__main__':
    unittest.main()

