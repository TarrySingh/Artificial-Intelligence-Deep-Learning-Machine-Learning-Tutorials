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
    using System.Collections.Generic;

    /** <summary>
     *  What does a tree look like?  ANTLR has a number of support classes
     *  such as CommonTreeNodeStream that work on these kinds of trees.  You
     *  don't have to make your trees implement this interface, but if you do,
     *  you'll be able to use more support code.
     *  </summary>
     *
     *  <remarks>
     *  NOTE: When constructing trees, ANTLR can build any kind of tree; it can
     *  even use Token objects as trees if you add a child list to your tokens.
     *
     *  This is a tree node without any payload; just navigation and factory stuff.
     *  </remarks>
     */
    public interface ITree
    {

        ITree GetChild( int i );

        int ChildCount
        {
            get;
        }

        // Tree tracks parent and child index now > 3.0

        ITree Parent
        {
            get;
            set;
        }

        /** <summary>Is there is a node above with token type ttype?</summary> */
        bool HasAncestor( int ttype );

        /** <summary>Walk upwards and get first ancestor with this token type.</summary> */
        ITree GetAncestor( int ttype );

        /** <summary>
         *  Return a list of all ancestors of this node.  The first node of
         *  list is the root and the last is the parent of this node.
         *  </summary>
         */
        IList<ITree> GetAncestors();

        /** <summary>This node is what child index? 0..n-1</summary> */
        int ChildIndex
        {
            get;
            set;
        }

        /** <summary>Set the parent and child index values for all children</summary> */
        void FreshenParentAndChildIndexes();

        /** <summary>
         *  Add t as a child to this node.  If t is null, do nothing.  If t
         *  is nil, add all children of t to this' children.
         *  </summary>
         */
        void AddChild( ITree t );

        /** <summary>Set ith child (0..n-1) to t; t must be non-null and non-nil node</summary> */
        void SetChild( int i, ITree t );

        object DeleteChild( int i );

        /** <summary>
         *  Delete children from start to stop and replace with t even if t is
         *  a list (nil-root tree).  num of children can increase or decrease.
         *  For huge child lists, inserting children can force walking rest of
         *  children to set their childindex; could be slow.
         *  </summary>
         */
        void ReplaceChildren( int startChildIndex, int stopChildIndex, object t );

        /** <summary>
         *  Indicates the node is a nil node but may still have children, meaning
         *  the tree is a flat list.
         *  </summary>
         */
        bool IsNil
        {
            get;
        }

        /** <summary>
         *  What is the smallest token index (indexing from 0) for this node
         *  and its children?
         *  </summary>
         */
        int TokenStartIndex
        {
            get;
            set;
        }

        /** <summary>
         *  What is the largest token index (indexing from 0) for this node
         *  and its children?
         *  </summary>
         */
        int TokenStopIndex
        {
            get;
            set;
        }

        ITree DupNode();

        /** <summary>Return a token type; needed for tree parsing</summary> */
        int Type
        {
            get;
        }

        string Text
        {
            get;
        }

        /** <summary>In case we don't have a token payload, what is the line for errors?</summary> */
        int Line
        {
            get;
        }

        int CharPositionInLine
        {
            get;
        }

        string ToStringTree();

        string ToString();
    }
}
