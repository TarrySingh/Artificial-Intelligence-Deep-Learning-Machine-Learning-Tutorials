import unittest
import textwrap
import antlr3
import antlr3.tree
import antlr3.debug
import testbase
import sys
import threading
import socket
import errno
import time

class Debugger(threading.Thread):
    def __init__(self, port):
        super(Debugger, self).__init__()
        self.events = []
        self.success = False
        self.port = port

    def run(self):
        # create listening socket
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(('127.0.0.1', self.port))
                break
            except socket.error, exc:
                if exc.args[0] != errno.ECONNREFUSED:
                    raise
                time.sleep(0.1)

        s.setblocking(1)

        output = s.makefile('w', 0)
        input = s.makefile('r', 0)

        # handshake
        l = input.readline().strip()
        assert l == 'ANTLR 2'
        l = input.readline().strip()
        assert l.startswith('grammar "')

        output.write('ACK\n')
        output.flush()

        while True:
            event = input.readline().strip()
            self.events.append(event.split('\t'))

            output.write('ACK\n')
            output.flush()

            if event == 'terminate':
                break

        s.close()
        self.success = True


class T(testbase.ANTLRTest):
    def execParser(self, grammar, grammarEntry, input, listener,
                   parser_args={}):
        if listener is None:
            port = 49100
            debugger = Debugger(port)
            debugger.start()
            # TODO(pink): install alarm, so it doesn't hang forever in case of a bug

        else:
            port = None

        try:
            lexerCls, parserCls = self.compileInlineGrammar(
                grammar, options='-debug')

            cStream = antlr3.StringStream(input)
            lexer = lexerCls(cStream)
            tStream = antlr3.CommonTokenStream(lexer)
            parser = parserCls(tStream, dbg=listener, port=port, **parser_args)
            getattr(parser, grammarEntry)()
    
        finally:
            if listener is None:
                debugger.join()
                return debugger

    def testBasicParser(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID EOF;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        listener = antlr3.debug.RecordDebugEventListener()

        self.execParser(
            grammar, 'a',
            input="a",
            listener=listener)

        # We only check that some LT events are present. How many is subject
        # to change (at the time of writing there are two, which is one too
        # many).
        lt_events = [event for event in listener.events
                     if event.startswith("LT ")]
        self.assertNotEqual(lt_events, [])

        # For the rest, filter out LT events to get a reliable test.
        expected = ["enterRule a",
                    "location 6:1",
                    "location 6:5",
                    "location 6:8",
                    "location 6:11",
                    "exitRule a"]
        found = [event for event in listener.events
                 if not event.startswith("LT ")]
        self.assertEqual(found, expected)

    def testSocketProxy(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID EOF;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['location', '6', '8'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                    ['location', '6', '11'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)

    def testRecognitionException(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID EOF;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a b",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['consumeHiddenToken', '1', '5', '99', '1', '1', '"'],
                    ['location', '6', '8'],
                    ['LT', '1', '2', '4', '0', '1', '2', '"b'],
                    ['LT', '1', '2', '4', '0', '1', '2', '"b'],
                    ['LT', '2', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '2', '4', '0', '1', '2', '"b'],
                    ['LT', '1', '2', '4', '0', '1', '2', '"b'],
                    ['beginResync'],
                    ['consumeToken', '2', '4', '0', '1', '2', '"b'],
                    ['endResync'],
                    ['exception', 'UnwantedTokenException', '2', '1', '2'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                    ['location', '6', '11'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testSemPred(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : {True}? ID EOF;
        ID : 'a'..'z'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['semanticPredicate', '1', 'True'],
                    ['location', '6', '13'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['location', '6', '16'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                    ['location', '6', '19'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testPositiveClosureBlock(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID ( ID | INT )+ EOF;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a 1 b c 3",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['consumeHiddenToken', '1', '6', '99', '1', '1', '"'],
                    ['location', '6', '8'],
                    ['enterSubRule', '1'],
                    ['enterDecision', '1'],
                    ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                    ['consumeToken', '2', '5', '0', '1', '2', '"1'],
                    ['consumeHiddenToken', '3', '6', '99', '1', '3', '"'],
                    ['enterDecision', '1'],
                    ['LT', '1', '4', '4', '0', '1', '4', '"b'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '4', '4', '0', '1', '4', '"b'],
                    ['consumeToken', '4', '4', '0', '1', '4', '"b'],
                    ['consumeHiddenToken', '5', '6', '99', '1', '5', '"'],
                    ['enterDecision', '1'],
                    ['LT', '1', '6', '4', '0', '1', '6', '"c'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '6', '4', '0', '1', '6', '"c'],
                    ['consumeToken', '6', '4', '0', '1', '6', '"c'],
                    ['consumeHiddenToken', '7', '6', '99', '1', '7', '"'],
                    ['enterDecision', '1'],
                    ['LT', '1', '8', '5', '0', '1', '8', '"3'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '8', '5', '0', '1', '8', '"3'],
                    ['consumeToken', '8', '5', '0', '1', '8', '"3'],
                    ['enterDecision', '1'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['exitDecision', '1'],
                    ['exitSubRule', '1'],
                    ['location', '6', '22'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                    ['location', '6', '25'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testClosureBlock(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID ( ID | INT )* EOF;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a 1 b c 3",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['consumeHiddenToken', '1', '6', '99', '1', '1', '"'],
                    ['location', '6', '8'],
                    ['enterSubRule', '1'],
                    ['enterDecision', '1'],
                    ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                    ['consumeToken', '2', '5', '0', '1', '2', '"1'],
                    ['consumeHiddenToken', '3', '6', '99', '1', '3', '"'],
                    ['enterDecision', '1'],
                    ['LT', '1', '4', '4', '0', '1', '4', '"b'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '4', '4', '0', '1', '4', '"b'],
                    ['consumeToken', '4', '4', '0', '1', '4', '"b'],
                    ['consumeHiddenToken', '5', '6', '99', '1', '5', '"'],
                    ['enterDecision', '1'],
                    ['LT', '1', '6', '4', '0', '1', '6', '"c'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '6', '4', '0', '1', '6', '"c'],
                    ['consumeToken', '6', '4', '0', '1', '6', '"c'],
                    ['consumeHiddenToken', '7', '6', '99', '1', '7', '"'],
                    ['enterDecision', '1'],
                    ['LT', '1', '8', '5', '0', '1', '8', '"3'],
                    ['exitDecision', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '8'],
                    ['LT', '1', '8', '5', '0', '1', '8', '"3'],
                    ['consumeToken', '8', '5', '0', '1', '8', '"3'],
                    ['enterDecision', '1'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['exitDecision', '1'],
                    ['exitSubRule', '1'],
                    ['location', '6', '22'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                    ['location', '6', '25'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testMismatchedSetException(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID ( ID | INT ) EOF;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['location', '6', '8'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['exception', 'MismatchedSetException', '1', '0', '-1'],
                    ['exception', 'MismatchedSetException', '1', '0', '-1'],
                    ['beginResync'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['endResync'],
                    ['location', '6', '24'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testBlock(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID ( b | c ) EOF;
        b : ID;
        c : INT;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a 1",
            listener=None)

        self.assertTrue(debugger.success)
        expected =  [['enterRule', 'T.g', 'a'],
                     ['location', '6', '1'],
                     ['enterAlt', '1'],
                     ['location', '6', '5'],
                     ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                     ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                     ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                     ['consumeHiddenToken', '1', '6', '99', '1', '1', '"'],
                     ['location', '6', '8'],
                     ['enterSubRule', '1'],
                     ['enterDecision', '1'],
                     ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                     ['exitDecision', '1'],
                     ['enterAlt', '2'],
                     ['location', '6', '14'],
                     ['enterRule', 'T.g', 'c'],
                     ['location', '8', '1'],
                     ['enterAlt', '1'],
                     ['location', '8', '5'],
                     ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                     ['LT', '1', '2', '5', '0', '1', '2', '"1'],
                     ['consumeToken', '2', '5', '0', '1', '2', '"1'],
                     ['location', '8', '8'],
                     ['exitRule', 'T.g', 'c'],
                     ['exitSubRule', '1'],
                     ['location', '6', '18'],
                     ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                     ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                     ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                     ['location', '6', '21'],
                     ['exitRule', 'T.g', 'a'],
                     ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testNoViableAlt(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ID ( b | c ) EOF;
        b : ID;
        c : INT;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        BANG : '!' ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a !",
            listener=None)

        self.assertTrue(debugger.success)
        expected =  [['enterRule', 'T.g', 'a'],
                     ['location', '6', '1'],
                     ['enterAlt', '1'],
                     ['location', '6', '5'],
                     ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                     ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                     ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                     ['consumeHiddenToken', '1', '7', '99', '1', '1', '"'],
                     ['location', '6', '8'],
                     ['enterSubRule', '1'],
                     ['enterDecision', '1'],
                     ['LT', '1', '2', '6', '0', '1', '2', '"!'],
                     ['LT', '1', '2', '6', '0', '1', '2', '"!'],
                     ['LT', '1', '2', '6', '0', '1', '2', '"!'],
                     ['exception', 'NoViableAltException', '2', '1', '2'],
                     ['exitDecision', '1'],
                     ['exitSubRule', '1'],
                     ['exception', 'NoViableAltException', '2', '1', '2'],
                     ['beginResync'],
                     ['LT', '1', '2', '6', '0', '1', '2', '"!'],
                     ['consumeToken', '2', '6', '0', '1', '2', '"!'],
                     ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                     ['endResync'],
                     ['location', '6', '21'],
                     ['exitRule', 'T.g', 'a'],
                     ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testRuleBlock(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : b | c;
        b : ID;
        c : INT;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="1",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterDecision', '1'],
                    ['LT', '1', '0', '5', '0', '1', '0', '"1'],
                    ['exitDecision', '1'],
                    ['enterAlt', '2'],
                    ['location', '6', '9'],
                    ['enterRule', 'T.g', 'c'],
                    ['location', '8', '1'],
                    ['enterAlt', '1'],
                    ['location', '8', '5'],
                    ['LT', '1', '0', '5', '0', '1', '0', '"1'],
                    ['LT', '1', '0', '5', '0', '1', '0', '"1'],
                    ['consumeToken', '0', '5', '0', '1', '0', '"1'],
                    ['location', '8', '8'],
                    ['exitRule', 'T.g', 'c'],
                    ['location', '6', '10'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testRuleBlockSingleAlt(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : b;
        b : ID;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['enterRule', 'T.g', 'b'],
                    ['location', '7', '1'],
                    ['enterAlt', '1'],
                    ['location', '7', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['location', '7', '7'],
                    ['exitRule', 'T.g', 'b'],
                    ['location', '6', '6'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testBlockSingleAlt(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ( b );
        b : ID;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['enterAlt', '1'],
                    ['location', '6', '7'],
                    ['enterRule', 'T.g', 'b'],
                    ['location', '7', '1'],
                    ['enterAlt', '1'],
                    ['location', '7', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['location', '7', '7'],
                    ['exitRule', 'T.g', 'b'],
                    ['location', '6', '10'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testDFA(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
        }
        a : ( b | c ) EOF;
        b : ID* INT;
        c : ID+ BANG;
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        BANG : '!';
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        debugger = self.execParser(
            grammar, 'a',
            input="a!",
            listener=None)

        self.assertTrue(debugger.success)
        expected = [['enterRule', 'T.g', 'a'],
                    ['location', '6', '1'],
                    ['enterAlt', '1'],
                    ['location', '6', '5'],
                    ['enterSubRule', '1'],
                    ['enterDecision', '1'],
                    ['mark', '0'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '1', '6', '0', '1', '1', '"!'],
                    ['consumeToken', '1', '6', '0', '1', '1', '"!'],
                    ['rewind', '0'],
                    ['exitDecision', '1'],
                    ['enterAlt', '2'],
                    ['location', '6', '11'],
                    ['enterRule', 'T.g', 'c'],
                    ['location', '8', '1'],
                    ['enterAlt', '1'],
                    ['location', '8', '5'],
                    ['enterSubRule', '3'],
                    ['enterDecision', '3'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['exitDecision', '3'],
                    ['enterAlt', '1'],
                    ['location', '8', '5'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['LT', '1', '0', '4', '0', '1', '0', '"a'],
                    ['consumeToken', '0', '4', '0', '1', '0', '"a'],
                    ['enterDecision', '3'],
                    ['LT', '1', '1', '6', '0', '1', '1', '"!'],
                    ['exitDecision', '3'],
                    ['exitSubRule', '3'],
                    ['location', '8', '9'],
                    ['LT', '1', '1', '6', '0', '1', '1', '"!'],
                    ['LT', '1', '1', '6', '0', '1', '1', '"!'],
                    ['consumeToken', '1', '6', '0', '1', '1', '"!'],
                    ['location', '8', '13'],
                    ['exitRule', 'T.g', 'c'],
                    ['exitSubRule', '1'],
                    ['location', '6', '15'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['LT', '1', '-1', '-1', '0', '0', '-1', '"'],
                    ['consumeToken', '-1', '-1', '0', '0', '-1', '"'],
                    ['location', '6', '18'],
                    ['exitRule', 'T.g', 'a'],
                    ['terminate']]

        self.assertEqual(debugger.events, expected)


    def testBasicAST(self):
        grammar = textwrap.dedent(
        r'''
        grammar T;
        options {
            language=Python;
            output=AST;
        }
        a : ( b | c ) EOF!;
        b : ID* INT -> ^(INT ID*);
        c : ID+ BANG -> ^(BANG ID+);
        ID : 'a'..'z'+ ;
        INT : '0'..'9'+ ;
        BANG : '!';
        WS : (' '|'\n') {$channel=HIDDEN;} ;
        ''')

        listener = antlr3.debug.RecordDebugEventListener()

        self.execParser(
            grammar, 'a',
            input="a!",
            listener=listener)

        # don't check output for now (too dynamic), I'm satisfied if it
        # doesn't crash


if __name__ == '__main__':
    unittest.main()
