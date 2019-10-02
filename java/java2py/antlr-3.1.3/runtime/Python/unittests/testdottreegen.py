# -*- coding: utf-8 -*-

import os
import unittest
from StringIO import StringIO
import textwrap

import stringtemplate3

from antlr3.dottreegen import toDOT
from antlr3.treewizard import TreeWizard
from antlr3.tree import CommonTreeAdaptor


class TestToDOT(unittest.TestCase):
    """Test case for the toDOT function."""

    def setUp(self):
        self.adaptor = CommonTreeAdaptor()
        self.tokens = [
            "", "", "", "", "", "A", "B", "C", "D", "E", "ID", "VAR"
            ]
        self.wiz = TreeWizard(self.adaptor, self.tokens)


    def testNone(self):
        """toDOT()"""

        treeST = stringtemplate3.StringTemplate(
            template=(
            "digraph {\n" +
            "  $nodes$\n" +
            "  $edges$\n" +
            "}\n")
            )

        edgeST = stringtemplate3.StringTemplate(
            template="$parent$ -> $child$\n"
            )

        tree = self.wiz.create("(A B (B C C) (B (C D D)))")
        st = toDOT(tree, self.adaptor, treeST, edgeST)

        result = st.toString()
        expected = textwrap.dedent(
            '''\
            digraph {
              n0 [label="A"];
              n1 [label="B"];
              n2 [label="B"];
              n3 [label="C"];
              n4 [label="C"];
              n5 [label="B"];
              n6 [label="C"];
              n7 [label="D"];
              n8 [label="D"];

              n0 -> n1
              n0 -> n2
              n2 -> n3
              n2 -> n4
              n0 -> n5
              n5 -> n6
              n6 -> n7
              n6 -> n8

            }
            '''
            )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
