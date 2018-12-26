import unittest
import textwrap
import antlr3
import antlr3.tree
import testbase

class T(testbase.ANTLRTest):
    def walkerClass(self, base):
        class TWalker(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self.traces = []


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise
            
        return TWalker
    

    def setUp(self):
        self.compileGrammar()
        self.compileGrammar('t047treeparserWalker.g', options='-trace')

        
    def testWalker(self):
        input = textwrap.dedent(
            '''\
            char c;
            int x;

            void bar(int x);

            int foo(int y, char d) {
              int i;
              for (i=0; i<3; i=i+1) {
                x=3;
                y=5;
              }
            }
            ''')
        
        cStream = antlr3.StringStream(input)
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        r = parser.program()

        self.failUnlessEqual(
            r.tree.toStringTree(),
            "(VAR_DEF char c) (VAR_DEF int x) (FUNC_DECL (FUNC_HDR void bar (ARG_DEF int x))) (FUNC_DEF (FUNC_HDR int foo (ARG_DEF int y) (ARG_DEF char d)) (BLOCK (VAR_DEF int i) (for (= i 0) (< i 3) (= i (+ i 1)) (BLOCK (= x 3) (= y 5)))))"
            )
        
        nodes = antlr3.tree.CommonTreeNodeStream(r.tree)
        nodes.setTokenStream(tStream)
        walker = self.getWalker(nodes)
        walker.program()

        # FIXME: need to crosscheck with Java target (compile walker with
        # -trace option), if this is the real list. For now I'm happy that
        # it does not crash ;)
        self.failUnlessEqual(
            walker.traces,
            [ '>program', '>declaration', '>variable', '>type', '<type',
              '>declarator', '<declarator', '<variable', '<declaration',
              '>declaration', '>variable', '>type', '<type', '>declarator',
              '<declarator', '<variable', '<declaration', '>declaration',
              '>functionHeader', '>type', '<type', '>formalParameter',
              '>type', '<type', '>declarator', '<declarator',
              '<formalParameter', '<functionHeader', '<declaration',
              '>declaration', '>functionHeader', '>type', '<type',
              '>formalParameter', '>type', '<type', '>declarator',
              '<declarator', '<formalParameter', '>formalParameter', '>type',
              '<type', '>declarator', '<declarator', '<formalParameter',
              '<functionHeader', '>block', '>variable', '>type', '<type',
              '>declarator', '<declarator', '<variable', '>stat', '>forStat',
              '>expr', '>expr', '>atom', '<atom', '<expr', '<expr', '>expr',
              '>expr', '>atom', '<atom', '<expr', '>expr', '>atom', '<atom',
              '<expr', '<expr', '>expr', '>expr', '>expr', '>atom', '<atom',
              '<expr', '>expr', '>atom', '<atom', '<expr', '<expr', '<expr',
              '>block', '>stat', '>expr', '>expr', '>atom', '<atom', '<expr',
              '<expr', '<stat', '>stat', '>expr', '>expr', '>atom', '<atom',
              '<expr', '<expr', '<stat', '<block', '<forStat', '<stat',
              '<block', '<declaration', '<program'
              ]
            )

    def testRuleLabelPropertyRefText(self):
        self.compileGrammar()
        self.compileGrammar('t047treeparserWalker.g', options='-trace')

        input = textwrap.dedent(
            '''\
            char c;
            ''')
        
        cStream = antlr3.StringStream(input)
        lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = self.getParser(tStream)
        r = parser.variable()

        nodes = antlr3.tree.CommonTreeNodeStream(r.tree)
        nodes.setTokenStream(tStream)
        walker = self.getWalker(nodes)
        r = walker.variable()

        self.failUnlessEqual(r, 'c')
        

if __name__ == '__main__':
    unittest.main()
