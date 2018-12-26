import unittest
import textwrap
import antlr3
import antlr3.tree
import stringtemplate3
import testbase
import sys
import os
from StringIO import StringIO

class T(testbase.ANTLRTest):
    def execParser(self, grammar, grammarEntry, input, group=None):
        lexerCls, parserCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        if group is not None:
            parser.templateLib = group
        result = getattr(parser, grammarEntry)()
        if result.st is not None:
            return result.st.toString()
        return None
    

    def testInlineTemplate(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a : ID INT
              -> template(id={$ID.text}, int={$INT.text})
                 "id=<id>, int=<int>"
            ;
            
            ID : 'a'..'z'+;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("id=abc, int=34", found)


    def testExternalTemplate(self):
        templates = textwrap.dedent(
            '''\
            group T;
            expr(args, op) ::= <<
            [<args; separator={<op>}>]
            >>
            '''
            )

        group = stringtemplate3.StringTemplateGroup(
            file=StringIO(templates),
            lexer='angle-bracket'
            )
        
        grammar = textwrap.dedent(
            r'''grammar T2;
            options {
              language=Python;
              output=template;
            }
            a : r+=arg OP r+=arg
              -> expr(op={$OP.text}, args={$r})
            ;
            arg: ID -> template(t={$ID.text}) "<t>";
            
            ID : 'a'..'z'+;
            OP: '+';
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "a + b",
            group
            )

        self.failUnlessEqual("[a+b]", found)


    def testEmptyTemplate(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a : ID INT
              -> 
            ;
            
            ID : 'a'..'z'+;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc 34"
            )

        self.failUnless(found is None)


    def testList(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a: (r+=b)* EOF
              -> template(r={$r})
                 "<r; separator=\",\">"
            ;

            b: ID
              -> template(t={$ID.text}) "<t>"
            ;
            
            ID : 'a'..'z'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc def ghi"
            )

        self.failUnlessEqual("abc,def,ghi", found)


    def testAction(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a: ID
              -> { stringtemplate3.StringTemplate("hello") }
            ;
            
            ID : 'a'..'z'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc"
            )

        self.failUnlessEqual("hello", found)


    def testTemplateExpressionInAction(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a: ID
              { $st = %{"hello"} }
            ;
            
            ID : 'a'..'z'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc"
            )

        self.failUnlessEqual("hello", found)


    def testTemplateExpressionInAction2(self):
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a: ID
              {
                res = %{"hello <foo>"}
                %res.foo = "world";
              }
              -> { res }
            ;
            
            ID : 'a'..'z'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc"
            )

        self.failUnlessEqual("hello world", found)


    @testbase.broken("Grammar parse fails, must be fixed by Ter", RuntimeError)
    def testIndirectTemplateConstructor(self):
        templates = textwrap.dedent(
            '''\
            group T;
            expr(args, op) ::= <<
            [<args; separator={<op>}>]
            >>
            '''
            )

        group = stringtemplate3.StringTemplateGroup(
            file=StringIO(templates),
            lexer='angle-bracket'
            )
        
        grammar = textwrap.dedent(
            r'''grammar T;
            options {
              language=Python;
              output=template;
            }
            a: ID
              {
                $st = %({"expr"})(args={[1, 2, 3]}, op={"+"})
              }
            ;
            
            ID : 'a'..'z'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc",
            group
            )

        self.failUnlessEqual("[1+2+3]", found)


    def testPredicates(self):
        grammar = textwrap.dedent(
            r'''grammar T3;
            options {
              language=Python;
              output=template;
            }
            a : ID INT
              -> {$ID.text=='a'}? template(int={$INT.text})
                                  "A: <int>"
              -> {$ID.text=='b'}? template(int={$INT.text})
                                  "B: <int>"
              ->                  template(int={$INT.text})
                                  "C: <int>"
            ;
            
            ID : 'a'..'z'+;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "b 34"
            )

        self.failUnlessEqual("B: 34", found)


    def testBacktrackingMode(self):
        grammar = textwrap.dedent(
            r'''grammar T4;
            options {
              language=Python;
              output=template;
              backtrack=true;
            }
            a : (ID INT)=> ID INT
              -> template(id={$ID.text}, int={$INT.text})
                 "id=<id>, int=<int>"
            ;
            
            ID : 'a'..'z'+;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            '''
            )

        found = self.execParser(
            grammar, 'a',
            "abc 34"
            )

        self.failUnlessEqual("id=abc, int=34", found)


    def testRewrite(self):
        grammar = textwrap.dedent(
            r'''grammar T5;
            options {
              language=Python;
              output=template;
              rewrite=true;
            }

            prog: stat+;

            stat
                : 'if' '(' expr ')' stat
                | 'return' return_expr ';'
                | '{' stat* '}'
                | ID '=' expr ';'
                ;

            return_expr
                : expr
                  -> template(t={$text}) <<boom(<t>)>>
                ;
                
            expr
                : ID
                | INT
                ;
                
            ID:  'a'..'z'+;
            INT: '0'..'9'+;
            WS: (' '|'\n')+ {$channel=HIDDEN;} ;
            COMMENT: '/*' (options {greedy=false;} : .)* '*/' {$channel = HIDDEN;} ;
            '''
            )

        input = textwrap.dedent(
            '''\
            if ( foo ) {
              b = /* bla */ 2;
              return 1 /* foo */;
            }

            /* gnurz */
            return 12;
            '''
            )
        
        lexerCls, parserCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.TokenRewriteStream(lexer)
        parser = parserCls(tStream)
        result = parser.prog()

        found = tStream.toString()
        
        expected = textwrap.dedent(
            '''\
            if ( foo ) {
              b = /* bla */ 2;
              return boom(1) /* foo */;
            }

            /* gnurz */
            return boom(12);
            '''
            )
        
        self.failUnlessEqual(expected, found)


    def testTreeRewrite(self):
        grammar = textwrap.dedent(
            r'''grammar T6;
            options {
              language=Python;
              output=AST;
            }

            tokens {
              BLOCK;
              ASSIGN;
            }
            
            prog: stat+;

            stat
                : IF '(' e=expr ')' s=stat
                  -> ^(IF $e $s)
                | RETURN expr ';'
                  -> ^(RETURN expr)                
                | '{' stat* '}'
                  -> ^(BLOCK stat*)                
                | ID '=' expr ';'
                  -> ^(ASSIGN ID expr)
                ;
                
            expr
                : ID
                | INT
                ;

            IF: 'if';
            RETURN: 'return';
            ID:  'a'..'z'+;
            INT: '0'..'9'+;
            WS: (' '|'\n')+ {$channel=HIDDEN;} ;
            COMMENT: '/*' (options {greedy=false;} : .)* '*/' {$channel = HIDDEN;} ;
            '''
            )

        treeGrammar = textwrap.dedent(
            r'''tree grammar T6Walker;
            options {
              language=Python;
              tokenVocab=T6;
              ASTLabelType=CommonTree;
              output=template;
              rewrite=true;
            }

            prog: stat+;

            stat
                : ^(IF expr stat)
                | ^(RETURN return_expr)                
                | ^(BLOCK stat*)                
                | ^(ASSIGN ID expr)
                ;

            return_expr
                : expr
                  -> template(t={$text}) <<boom(<t>)>>
                ;
            
            expr
                : ID
                | INT
                ;
            '''
            )

        input = textwrap.dedent(
            '''\
            if ( foo ) {
              b = /* bla */ 2;
              return 1 /* foo */;
            }

            /* gnurz */
            return 12;
            '''
            )
        
        lexerCls, parserCls = self.compileInlineGrammar(grammar)
        walkerCls = self.compileInlineGrammar(treeGrammar)
        
        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.TokenRewriteStream(lexer)
        parser = parserCls(tStream)
        tree = parser.prog().tree
        nodes = antlr3.tree.CommonTreeNodeStream(tree)
        nodes.setTokenStream(tStream)
        walker = walkerCls(nodes)
        walker.prog()
        
        found = tStream.toString()
        
        expected = textwrap.dedent(
            '''\
            if ( foo ) {
              b = /* bla */ 2;
              return boom(1) /* foo */;
            }

            /* gnurz */
            return boom(12);
            '''
            )
        
        self.failUnlessEqual(expected, found)

        
if __name__ == '__main__':
    unittest.main()
