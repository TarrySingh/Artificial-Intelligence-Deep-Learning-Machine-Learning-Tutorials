import unittest
import textwrap
import antlr3
import antlr3.tree
import testbase
import sys

class TestAutoAST(testbase.ANTLRTest):
    def parserClass(self, base):
        class TParser(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self._errors = []
                self._output = ""


            def capture(self, t):
                self._output += t


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def emitErrorMessage(self, msg):
                self._errors.append(msg)

            
        return TParser
    

    def lexerClass(self, base):
        class TLexer(base):
            def __init__(self, *args, **kwargs):
                base.__init__(self, *args, **kwargs)

                self._output = ""


            def capture(self, t):
                self._output += t


            def traceIn(self, ruleName, ruleIndex):
                self.traces.append('>'+ruleName)


            def traceOut(self, ruleName, ruleIndex):
                self.traces.append('<'+ruleName)


            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise
            
        return TLexer
    

    def execParser(self, grammar, grammarEntry, input, expectErrors=False):
        lexerCls, parserCls = self.compileInlineGrammar(grammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        r = getattr(parser, grammarEntry)()

        if not expectErrors:
            self.assertEquals(len(parser._errors), 0, parser._errors)

        result = ""

        if r is not None:
            if hasattr(r, 'result'):
                result += r.result

            if r.tree is not None:
                result += r.tree.toStringTree()

        if not expectErrors:
            return result

        else:
            return result, parser._errors
        

    def execTreeParser(self, grammar, grammarEntry, treeGrammar, treeEntry, input):
        lexerCls, parserCls = self.compileInlineGrammar(grammar)
        walkerCls = self.compileInlineGrammar(treeGrammar)

        cStream = antlr3.StringStream(input)
        lexer = lexerCls(cStream)
        tStream = antlr3.CommonTokenStream(lexer)
        parser = parserCls(tStream)
        r = getattr(parser, grammarEntry)()
        nodes = antlr3.tree.CommonTreeNodeStream(r.tree)
        nodes.setTokenStream(tStream)
        walker = walkerCls(nodes)
        r = getattr(walker, treeEntry)()

        if r is not None:
            return r.tree.toStringTree()

        return ""


    def testTokenList(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;};
            ''')
        
        found = self.execParser(grammar, "a", "abc 34")
        self.assertEquals("abc 34", found);


    def testTokenListInSingleAltBlock(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : (ID INT) ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(grammar,"a", "abc 34")
        self.assertEquals("abc 34", found)


    def testSimpleRootAtOuterLevel(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : ID^ INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
    
        found = self.execParser(grammar, "a", "abc 34")
        self.assertEquals("(abc 34)", found)


    def testSimpleRootAtOuterLevelReverse(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : INT ID^ ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
    
        found = self.execParser(grammar, "a", "34 abc")
        self.assertEquals("(abc 34)", found)


    def testBang(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID INT! ID! INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "abc 34 dag 4532")
        self.assertEquals("abc 4532", found)


    def testOptionalThenRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ( ID INT )? ID^ ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(grammar, "a", "a 1 b")
        self.assertEquals("(b a 1)", found)


    def testLabeledStringRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : v='void'^ ID ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "void foo;")
        self.assertEquals("(void foo ;)", found)


    def testWildcard(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : v='void'^ . ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(grammar, "a", "void foo;")
        self.assertEquals("(void foo ;)", found)


    def testWildcardRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : v='void' .^ ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "void foo;")
        self.assertEquals("(foo void ;)", found)


    def testWildcardRootWithLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : v='void' x=.^ ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "void foo;")
        self.assertEquals("(foo void ;)", found)


    def testWildcardRootWithListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : v='void' x=.^ ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "void foo;")
        self.assertEquals("(foo void ;)", found)


    def testWildcardBangWithListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : v='void' x=.! ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(grammar, "a", "void foo;")
        self.assertEquals("void ;", found)


    def testRootRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID^ INT^ ID ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a 34 c")
        self.assertEquals("(34 a c)", found)


    def testRootRoot2(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID INT^ ID^ ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a 34 c")
        self.assertEquals("(c (34 a))", found)


    def testRootThenRootInLoop(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID^ (INT '*'^ ID)+ ;
            ID  : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a 34 * b 9 * c")
        self.assertEquals("(* (* (a 34) b 9) c)", found)


    def testNestedSubrule(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : 'void' (({pass}ID|INT) ID | 'null' ) ';' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "void a b;")
        self.assertEquals("void a b ;", found)


    def testInvokeRule(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a  : type ID ;
            type : {pass}'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "int a")
        self.assertEquals("int a", found)


    def testInvokeRuleAsRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a  : type^ ID ;
            type : {pass}'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "int a")
        self.assertEquals("(int a)", found)


    def testInvokeRuleAsRootWithLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a  : x=type^ ID ;
            type : {pass}'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "int a")
        self.assertEquals("(int a)", found)


    def testInvokeRuleAsRootWithListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a  : x+=type^ ID ;
            type : {pass}'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "int a")
        self.assertEquals("(int a)", found)


    def testRuleRootInLoop(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID ('+'^ ID)* ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a+b+c+d")
        self.assertEquals("(+ (+ (+ a b) c) d)", found)


    def testRuleInvocationRuleRootInLoop(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID (op^ ID)* ;
            op : {pass}'+' | '-' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a+b+c-d")
        self.assertEquals("(- (+ (+ a b) c) d)", found)


    def testTailRecursion(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            s : a ;
            a : atom ('exp'^ a)? ;
            atom : INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "s", "3 exp 4 exp 5")
        self.assertEquals("(exp 3 (exp 4 5))", found)


    def testSet(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID|INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "abc")
        self.assertEquals("abc", found)


    def testSetRoot(self):
        grammar = textwrap.dedent(
        r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ('+' | '-')^ ID ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "+abc")
        self.assertEquals("(+ abc)", found)


    @testbase.broken("FAILS until antlr.g rebuilt in v3", RuntimeError)
    def testSetRootWithLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : x=('+' | '-')^ ID ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "+abc")
        self.assertEquals("(+ abc)", found)


    def testSetAsRuleRootInLoop(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ID (('+'|'-')^ ID)* ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a+b-c")
        self.assertEquals("(- (+ a b) c)", found)


    def testNotSet(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ~ID '+' INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "34+2")
        self.assertEquals("34 + 2", found)


    def testNotSetWithLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : x=~ID '+' INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "34+2")
        self.assertEquals("34 + 2", found)


    def testNotSetWithListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : x=~ID '+' INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "34+2")
        self.assertEquals("34 + 2", found)


    def testNotSetRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ~'+'^ INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "34 55")
        self.assertEquals("(34 55)", found)


    def testNotSetRootWithLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ~'+'^ INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "34 55")
        self.assertEquals("(34 55)", found)


    def testNotSetRootWithListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : ~'+'^ INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "34 55")
        self.assertEquals("(34 55)", found)


    def testNotSetRuleRootInLoop(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : INT (~INT^ INT)* ;
            blort : '+' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "3+4+5")
        self.assertEquals("(+ (+ 3 4) 5)", found)


    @testbase.broken("FIXME: What happened to the semicolon?", AssertionError)
    def testTokenLabelReuse(self):
        # check for compilation problem due to multiple defines
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result] : id=ID id=ID {$result = "2nd id="+$id.text+";";} ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b")
        self.assertEquals("2nd id=b;a b", found)


    def testTokenLabelReuse2(self):
        # check for compilation problem due to multiple defines
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result]: id=ID id=ID^ {$result = "2nd id="+$id.text+',';} ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b")
        self.assertEquals("2nd id=b,(b a)", found)


    def testTokenListLabelReuse(self):
        # check for compilation problem due to multiple defines
        # make sure ids has both ID tokens
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result] : ids+=ID ids+=ID {$result = "id list=["+",".join([t.text for t in $ids])+'],';} ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')

        found = self.execParser(grammar, "a", "a b")
        expecting = "id list=[a,b],a b"
        self.assertEquals(expecting, found)


    def testTokenListLabelReuse2(self):
        # check for compilation problem due to multiple defines
        # make sure ids has both ID tokens
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result] : ids+=ID^ ids+=ID {$result = "id list=["+",".join([t.text for t in $ids])+'],';} ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b")
        expecting = "id list=[a,b],(a b)"
        self.assertEquals(expecting, found)


    def testTokenListLabelRuleRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : id+=ID^ ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a")
        self.assertEquals("a", found)


    def testTokenListLabelBang(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : id+=ID! ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a")
        self.assertEquals("", found)


    def testRuleListLabel(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result]: x+=b x+=b {
            t=$x[1]
            $result = "2nd x="+t.toStringTree()+',';
            };
            b : ID;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b")
        self.assertEquals("2nd x=b,a b", found)


    def testRuleListLabelRuleRoot(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result] : ( x+=b^ )+ {
            $result = "x="+$x[1].toStringTree()+',';
            } ;
            b : ID;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b")
        self.assertEquals("x=(b a),(b a)", found)


    def testRuleListLabelBang(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a returns [result] : x+=b! x+=b {
            $result = "1st x="+$x[0].toStringTree()+',';
            } ;
            b : ID;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b")
        self.assertEquals("1st x=a,b", found)


    def testComplicatedMelange(self):
        # check for compilation problem
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options {language=Python;output=AST;}
            a : A b=B b=B c+=C c+=C D {s = $D.text} ;
            A : 'a' ;
            B : 'b' ;
            C : 'c' ;
            D : 'd' ;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "a b b c c d")
        self.assertEquals("a b b c c d", found)


    def testReturnValueWithAST(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a returns [result] : ID b { $result = str($b.i) + '\n';} ;
            b returns [i] : INT {$i=int($INT.text);} ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found = self.execParser(grammar, "a", "abc 34")
        self.assertEquals("34\nabc 34", found)


    def testSetLoop(self):
        grammar = textwrap.dedent(
            r'''
            grammar T;
            options { language=Python;output=AST; }
            r : (INT|ID)+ ; 
            ID : 'a'..'z' + ;
            INT : '0'..'9' +;
            WS: (' ' | '\n' | '\\t')+ {$channel = HIDDEN;};
            ''')
        
        found = self.execParser(grammar, "r", "abc 34 d")
        self.assertEquals("abc 34 d", found)


    def testExtraTokenInSimpleDecl(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            decl : type^ ID '='! INT ';'! ;
            type : 'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "decl", "int 34 x=1;",
                                        expectErrors=True)
        self.assertEquals(["line 1:4 extraneous input u'34' expecting ID"],
                          errors)
        self.assertEquals("(int x 1)", found) # tree gets correct x and 1 tokens


    def testMissingIDInSimpleDecl(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            tokens {EXPR;}
            decl : type^ ID '='! INT ';'! ;
            type : 'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "decl", "int =1;",
                                        expectErrors=True)
        self.assertEquals(["line 1:4 missing ID at u'='"], errors)
        self.assertEquals("(int <missing ID> 1)", found) # tree gets invented ID token


    def testMissingSetInSimpleDecl(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            tokens {EXPR;}
            decl : type^ ID '='! INT ';'! ;
            type : 'int' | 'float' ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "decl", "x=1;",
                                        expectErrors=True)
        self.assertEquals(["line 1:0 mismatched input u'x' expecting set None"], errors)
        self.assertEquals("(<error: x> x 1)", found) # tree gets invented ID token


    def testMissingTokenGivesErrorNode(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : ID INT ; // follow is EOF
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "a", "abc", expectErrors=True)
        self.assertEquals(["line 0:-1 missing INT at '<EOF>'"], errors)
        self.assertEquals("abc <missing INT>", found)


    def testMissingTokenGivesErrorNodeInInvokedRule(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : b ;
            b : ID INT ; // follow should see EOF
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
    
        found, errors = self.execParser(grammar, "a", "abc", expectErrors=True)
        self.assertEquals(["line 0:-1 mismatched input '<EOF>' expecting INT"], errors)
        self.assertEquals("<mismatched token: <EOF>, resync=abc>", found)


    def testExtraTokenGivesErrorNode(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : b c ;
            b : ID ;
            c : INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "a", "abc ick 34",
                                        expectErrors=True)
        self.assertEquals(["line 1:4 extraneous input u'ick' expecting INT"],
                          errors)
        self.assertEquals("abc 34", found)


    def testMissingFirstTokenGivesErrorNode(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : ID INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "a", "34", expectErrors=True)
        self.assertEquals(["line 1:0 missing ID at u'34'"], errors)
        self.assertEquals("<missing ID> 34", found)


    def testMissingFirstTokenGivesErrorNode2(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : b c ;
            b : ID ;
            c : INT ;
            ID : 'a'..'z'+ ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "a", "34", expectErrors=True)
        
        # finds an error at the first token, 34, and re-syncs.
        # re-synchronizing does not consume a token because 34 follows
        # ref to rule b (start of c). It then matches 34 in c.
        self.assertEquals(["line 1:0 missing ID at u'34'"], errors)
        self.assertEquals("<missing ID> 34", found)


    def testNoViableAltGivesErrorNode(self):
        grammar = textwrap.dedent(
            r'''
            grammar foo;
            options {language=Python;output=AST;}
            a : b | c ;
            b : ID ;
            c : INT ;
            ID : 'a'..'z'+ ;
            S : '*' ;
            INT : '0'..'9'+;
            WS : (' '|'\n') {$channel=HIDDEN;} ;
            ''')
        
        found, errors = self.execParser(grammar, "a", "*", expectErrors=True)
        self.assertEquals(["line 1:0 no viable alternative at input u'*'"],
                          errors)
        self.assertEquals("<unexpected: [@0,0:0=u'*',<6>,1:0], resync=*>",
                          found)


if __name__ == '__main__':
    unittest.main()
