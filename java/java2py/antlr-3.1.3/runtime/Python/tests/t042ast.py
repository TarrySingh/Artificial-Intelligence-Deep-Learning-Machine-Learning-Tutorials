import unittest
import textwrap
import antlr3
import testbase

class t042ast(testbase.ANTLRTest):
##     def lexerClass(self, base):
##         class TLexer(base):
##             def reportError(self, re):
##                 # no error recovery yet, just crash!
##                 raise re

##         return TLexer
    

    def parserClass(self, base):
        class TParser(base):
            def recover(self, input, re):
                # no error recovery yet, just crash!
                raise

        return TParser
    

    def parse(self, text, method, rArgs=[], **kwargs):
        self.compileGrammar() #options='-trace')
        
        cStream = antlr3.StringStream(text)
        self.lexer = self.getLexer(cStream)
        tStream = antlr3.CommonTokenStream(self.lexer)
        self.parser = self.getParser(tStream)
        
        for attr, val in kwargs.items():
            setattr(self.parser, attr, val)
            
        return getattr(self.parser, method)(*rArgs)

    
    def testR1(self):
        r = self.parse("1 + 2", 'r1')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(+ 1 2)'
            )


    def testR2a(self):
        r = self.parse("assert 2+3;", 'r2')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(assert (+ 2 3))'
            )


    def testR2b(self):
        r = self.parse("assert 2+3 : 5;", 'r2')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(assert (+ 2 3) 5)'
            )


    def testR3a(self):
        r = self.parse("if 1 fooze", 'r3')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(if 1 fooze)'
            )


    def testR3b(self):
        r = self.parse("if 1 fooze else fooze", 'r3')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(if 1 fooze fooze)'
            )


    def testR4a(self):
        r = self.parse("while 2 fooze", 'r4')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(while 2 fooze)'
            )


    def testR5a(self):
        r = self.parse("return;", 'r5')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'return'
            )


    def testR5b(self):
        r = self.parse("return 2+3;", 'r5')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(return (+ 2 3))'
            )


    def testR6a(self):
        r = self.parse("3", 'r6')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '3'
            )


    def testR6b(self):
        r = self.parse("3 a", 'r6')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '3 a'
            )


    def testR7(self):
        r = self.parse("3", 'r7')
        self.failUnless(
            r.tree is None
            )


    def testR8(self):
        r = self.parse("var foo:bool", 'r8')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(var bool foo)'
            )


    def testR9(self):
        r = self.parse("int foo;", 'r9')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(VARDEF int foo)'
            )


    def testR10(self):
        r = self.parse("10", 'r10')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '10.0'
            )


    def testR11a(self):
        r = self.parse("1+2", 'r11')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(EXPR (+ 1 2))'
            )


    def testR11b(self):
        r = self.parse("", 'r11')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'EXPR'
            )


    def testR12a(self):
        r = self.parse("foo", 'r12')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'foo'
            )


    def testR12b(self):
        r = self.parse("foo, bar, gnurz", 'r12')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'foo bar gnurz'
            )


    def testR13a(self):
        r = self.parse("int foo;", 'r13')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(int foo)'
            )


    def testR13b(self):
        r = self.parse("bool foo, bar, gnurz;", 'r13')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(bool foo bar gnurz)'
            )


    def testR14a(self):
        r = self.parse("1+2 int", 'r14')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(EXPR (+ 1 2) int)'
            )


    def testR14b(self):
        r = self.parse("1+2 int bool", 'r14')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(EXPR (+ 1 2) int bool)'
            )


    def testR14c(self):
        r = self.parse("int bool", 'r14')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(EXPR int bool)'
            )


    def testR14d(self):
        r = self.parse("fooze fooze int bool", 'r14')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(EXPR fooze fooze int bool)'
            )


    def testR14e(self):
        r = self.parse("7+9 fooze fooze int bool", 'r14')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(EXPR (+ 7 9) fooze fooze int bool)'
            )


    def testR15(self):
        r = self.parse("7", 'r15')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '7 7'
            )


    def testR16a(self):
        r = self.parse("int foo", 'r16')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(int foo)'
            )


    def testR16b(self):
        r = self.parse("int foo, bar, gnurz", 'r16')
            
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(int foo) (int bar) (int gnurz)'
            )


    def testR17a(self):
        r = self.parse("for ( fooze ; 1 + 2 ; fooze ) fooze", 'r17')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(for fooze (+ 1 2) fooze fooze)'
            )


    def testR18a(self):
        r = self.parse("for", 'r18')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'BLOCK'
            )


    def testR19a(self):
        r = self.parse("for", 'r19')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'for'
            )


    def testR20a(self):
        r = self.parse("for", 'r20')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'FOR'
            )


    def testR21a(self):
        r = self.parse("for", 'r21')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'BLOCK'
            )


    def testR22a(self):
        r = self.parse("for", 'r22')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'for'
            )


    def testR23a(self):
        r = self.parse("for", 'r23')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'FOR'
            )


    def testR24a(self):
        r = self.parse("fooze 1 + 2", 'r24')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(fooze (+ 1 2))'
            )


    def testR25a(self):
        r = self.parse("fooze, fooze2 1 + 2", 'r25')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(fooze (+ 1 2))'
            )


    def testR26a(self):
        r = self.parse("fooze, fooze2", 'r26')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(BLOCK fooze fooze2)'
            )


    def testR27a(self):
        r = self.parse("fooze 1 + 2", 'r27')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(fooze (fooze (+ 1 2)))'
            )
            

    def testR28(self):
        r = self.parse("foo28a", 'r28')
        self.failUnless(
            r.tree is None
            )


    def testR29(self):
        try:
            r = self.parse("", 'r29')
            self.fail()
        except RuntimeError:
            pass


# FIXME: broken upstream?
##     def testR30(self):
##         try:
##             r = self.parse("fooze fooze", 'r30')
##             self.fail(r.tree.toStringTree())
##         except RuntimeError:
##             pass


    def testR31a(self):
        r = self.parse("public int gnurz = 1 + 2;", 'r31', flag=0)
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(VARDEF gnurz public int (+ 1 2))'
            )


    def testR31b(self):
        r = self.parse("public int gnurz = 1 + 2;", 'r31', flag=1)
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(VARIABLE gnurz public int (+ 1 2))'
            )


    def testR31c(self):
        r = self.parse("public int gnurz = 1 + 2;", 'r31', flag=2)
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(FIELD gnurz public int (+ 1 2))'
            )


    def testR32a(self):
        r = self.parse("gnurz 32", 'r32', [1], flag=2)
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'gnurz'
            )


    def testR32b(self):
        r = self.parse("gnurz 32", 'r32', [2], flag=2)
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '32'
            )


    def testR32c(self):
        r = self.parse("gnurz 32", 'r32', [3], flag=2)
        self.failUnless(
            r.tree is None
            )


    def testR33a(self):
        r = self.parse("public private fooze", 'r33')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'fooze'
            )


    def testR34a(self):
        r = self.parse("public class gnurz { fooze fooze2 }", 'r34')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(class gnurz public fooze fooze2)'
            )


    def testR34b(self):
        r = self.parse("public class gnurz extends bool implements int, bool { fooze fooze2 }", 'r34')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(class gnurz public (extends bool) (implements int bool) fooze fooze2)'
            )


    def testR35(self):
        try:
            r = self.parse("{ extends }", 'r35')
            self.fail()
            
        except RuntimeError:
            pass


    def testR36a(self):
        r = self.parse("if ( 1 + 2 ) fooze", 'r36')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(if (EXPR (+ 1 2)) fooze)'
            )


    def testR36b(self):
        r = self.parse("if ( 1 + 2 ) fooze else fooze2", 'r36')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(if (EXPR (+ 1 2)) fooze fooze2)'
            )


    def testR37(self):
        r = self.parse("1 + 2 + 3", 'r37')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(+ (+ 1 2) 3)'
            )


    def testR38(self):
        r = self.parse("1 + 2 + 3", 'r38')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(+ (+ 1 2) 3)'
            )


    def testR39a(self):
        r = self.parse("gnurz[1]", 'r39')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(INDEX gnurz 1)'
            )


    def testR39b(self):
        r = self.parse("gnurz(2)", 'r39')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(CALL gnurz 2)'
            )


    def testR39c(self):
        r = self.parse("gnurz.gnarz", 'r39')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(FIELDACCESS gnurz gnarz)'
            )


    def testR39d(self):
        r = self.parse("gnurz.gnarz.gnorz", 'r39')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(FIELDACCESS (FIELDACCESS gnurz gnarz) gnorz)'
            )


    def testR40(self):
        r = self.parse("1 + 2 + 3;", 'r40')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(+ 1 2 3)'
            )


    def testR41(self):
        r = self.parse("1 + 2 + 3;", 'r41')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(3 (2 1))'
            )


    def testR42(self):
        r = self.parse("gnurz, gnarz, gnorz", 'r42')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'gnurz gnarz gnorz'
            )


    def testR43(self):
        r = self.parse("gnurz, gnarz, gnorz", 'r43')
        self.failUnless(
            r.tree is None
            )
        self.failUnlessEqual(
            r.res,
            ['gnurz', 'gnarz', 'gnorz']
            )


    def testR44(self):
        r = self.parse("gnurz, gnarz, gnorz", 'r44')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(gnorz (gnarz gnurz))'
            )


    def testR45(self):
        r = self.parse("gnurz", 'r45')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'gnurz'
            )


    def testR46(self):
        r = self.parse("gnurz, gnarz, gnorz", 'r46')
        self.failUnless(
            r.tree is None
            )
        self.failUnlessEqual(
            r.res,
            ['gnurz', 'gnarz', 'gnorz']
            )


    def testR47(self):
        r = self.parse("gnurz, gnarz, gnorz", 'r47')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'gnurz gnarz gnorz'
            )


    def testR48(self):
        r = self.parse("gnurz, gnarz, gnorz", 'r48')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'gnurz gnarz gnorz'
            )


    def testR49(self):
        r = self.parse("gnurz gnorz", 'r49')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(gnurz gnorz)'
            )


    def testR50(self):
        r = self.parse("gnurz", 'r50')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(1.0 gnurz)'
            )


    def testR51(self):
        r = self.parse("gnurza gnurzb gnurzc", 'r51')
        self.failUnlessEqual(
            r.res.toStringTree(),
            'gnurzb'
            )


    def testR52(self):
        r = self.parse("gnurz", 'r52')
        self.failUnlessEqual(
            r.res.toStringTree(),
            'gnurz'
            )


    def testR53(self):
        r = self.parse("gnurz", 'r53')
        self.failUnlessEqual(
            r.res.toStringTree(),
            'gnurz'
            )


    def testR54(self):
        r = self.parse("gnurza 1 + 2 gnurzb", 'r54')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(+ 1 2)'
            )


    def testR55a(self):
        r = self.parse("public private 1 + 2", 'r55')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'public private (+ 1 2)'
            )


    def testR55b(self):
        r = self.parse("public fooze", 'r55')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'public fooze'
            )


    def testR56(self):
        r = self.parse("a b c d", 'r56')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'foo'
            )


    def testR57(self):
        r = self.parse("a b c d", 'r57')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            'foo'
            )


    def testR59(self):
        r = self.parse("a b c fooze", 'r59')
        self.failUnlessEqual(
            r.tree.toStringTree(),
            '(a fooze) (b fooze) (c fooze)'
            )



if __name__ == '__main__':
    unittest.main()

