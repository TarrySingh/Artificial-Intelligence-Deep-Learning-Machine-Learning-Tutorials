/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008-2009 Sam Harwell, Pixel Mine, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

namespace Antlr.Runtime.Tree
{
    /** <summary>A stream of tree nodes, accessing nodes from a tree of some kind</summary> */
    public interface ITreeNodeStream : IIntStream
    {
        /** <summary>
         *  Get a tree node at an absolute index i; 0..n-1.
         *  If you don't want to buffer up nodes, then this method makes no
         *  sense for you.
         *  </summary>
         */
        object this[int i]
        {
            get;
        }

        /** <summary>
         *  Get tree node at current input pointer + i ahead where i=1 is next node.
         *  i&lt;0 indicates nodes in the past.  So LT(-1) is previous node, but
         *  implementations are not required to provide results for k &lt; -1.
         *  LT(0) is undefined.  For i&gt;=n, return null.
         *  Return null for LT(0) and any index that results in an absolute address
         *  that is negative.
         *  </summary>
         *
         *  <remarks>
         *  This is analogus to the LT() method of the TokenStream, but this
         *  returns a tree node instead of a token.  Makes code gen identical
         *  for both parser and tree grammars. :)
         *  </remarks>
         */
        object LT( int k );

        /** <summary>
         *  Where is this stream pulling nodes from?  This is not the name, but
         *  the object that provides node objects.
         *  </summary>
         */
        object TreeSource
        {
            get;
        }

        /** <summary>
         *  If the tree associated with this stream was created from a TokenStream,
         *  you can specify it here.  Used to do rule $text attribute in tree
         *  parser.  Optional unless you use tree parser rule text attribute
         *  or output=template and rewrite=true options.
         *  </summary>
         */
        ITokenStream TokenStream
        {
            get;
        }

        /** <summary>
         *  What adaptor can tell me how to interpret/navigate nodes and
         *  trees.  E.g., get text of a node.
         *  </summary>
         */
        ITreeAdaptor TreeAdaptor
        {
            get;
        }

        /** <summary>
         *  As we flatten the tree, we use UP, DOWN nodes to represent
         *  the tree structure.  When debugging we need unique nodes
         *  so we have to instantiate new ones.  When doing normal tree
         *  parsing, it's slow and a waste of memory to create unique
         *  navigation nodes.  Default should be false;
         *  </summary>
         */
        bool UniqueNavigationNodes
        {
            get;
            set;
        }

        /** <summary>
         *  Return the text of all nodes from start to stop, inclusive.
         *  If the stream does not buffer all the nodes then it can still
         *  walk recursively from start until stop.  You can always return
         *  null or "" too, but users should not access $ruleLabel.text in
         *  an action of course in that case.
         *  </summary>
         */
        string ToString( object start, object stop );


        #region REWRITING TREES (used by tree parser)

        /** <summary>
         *  Replace from start to stop child index of parent with t, which might
         *  be a list.  Number of children may be different
         *  after this call.  The stream is notified because it is walking the
         *  tree and might need to know you are monkeying with the underlying
         *  tree.  Also, it might be able to modify the node stream to avoid
         *  restreaming for future phases.
         *  </summary>
         *
         *  <remarks>
         *  If parent is null, don't do anything; must be at root of overall tree.
         *  Can't replace whatever points to the parent externally.  Do nothing.
         *  </remarks>
         */
        void ReplaceChildren( object parent, int startChildIndex, int stopChildIndex, object t );

        #endregion

    }
}
