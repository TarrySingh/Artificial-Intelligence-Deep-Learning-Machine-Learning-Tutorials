import unittest
import antlr3
import testbase


class TestRecognitionException(unittest.TestCase):
    """Tests for the antlr3.RecognitionException class"""

    def testInitNone(self):
        """RecognitionException.__init__()"""

        exc = antlr3.RecognitionException()
        

class TestEarlyExitException(unittest.TestCase):
    """Tests for the antlr3.EarlyExitException class"""

    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """EarlyExitException.__init__()"""

        exc = antlr3.EarlyExitException()
        

class TestFailedPredicateException(unittest.TestCase):
    """Tests for the antlr3.FailedPredicateException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """FailedPredicateException.__init__()"""

        exc = antlr3.FailedPredicateException()
        

class TestMismatchedNotSetException(unittest.TestCase):
    """Tests for the antlr3.MismatchedNotSetException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """MismatchedNotSetException.__init__()"""

        exc = antlr3.MismatchedNotSetException()
        

class TestMismatchedRangeException(unittest.TestCase):
    """Tests for the antlr3.MismatchedRangeException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """MismatchedRangeException.__init__()"""

        exc = antlr3.MismatchedRangeException()
        

class TestMismatchedSetException(unittest.TestCase):
    """Tests for the antlr3.MismatchedSetException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """MismatchedSetException.__init__()"""

        exc = antlr3.MismatchedSetException()
        

class TestMismatchedTokenException(unittest.TestCase):
    """Tests for the antlr3.MismatchedTokenException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """MismatchedTokenException.__init__()"""

        exc = antlr3.MismatchedTokenException()
        

class TestMismatchedTreeNodeException(unittest.TestCase):
    """Tests for the antlr3.MismatchedTreeNodeException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """MismatchedTreeNodeException.__init__()"""

        exc = antlr3.MismatchedTreeNodeException()
        

class TestNoViableAltException(unittest.TestCase):
    """Tests for the antlr3.NoViableAltException class"""
    
    @testbase.broken("FIXME", Exception)
    def testInitNone(self):
        """NoViableAltException.__init__()"""

        exc = antlr3.NoViableAltException()
        

if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
