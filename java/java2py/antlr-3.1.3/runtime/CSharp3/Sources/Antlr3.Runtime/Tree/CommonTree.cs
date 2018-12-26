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
    /** <summary>
     *  A tree node that is wrapper for a Token object.  After 3.0 release
     *  while building tree rewrite stuff, it became clear that computing
     *  parent and child index is very difficult and cumbersome.  Better to
     *  spend the space in every tree node.  If you don't want these extra
     *  fields, it's easy to cut them out in your own BaseTree subclass.
     *  </summary>
     */
    [System.Serializable]
    public class CommonTree : BaseTree
    {
        /** <summary>A single token is the payload</summary> */
        public IToken token;

        /** <summary>
         *  What token indexes bracket all tokens associated with this node
         *  and below?
         *  </summary>
         */
        protected int startIndex = -1;
        protected int stopIndex = -1;

        /** <summary>Who is the parent node of this node; if null, implies node is root</summary> */
        public CommonTree parent;

        /** <summary>What index is this node in the child list? Range: 0..n-1</summary> */
        public int childIndex = -1;

        public CommonTree()
        {
        }

        public CommonTree( CommonTree node )
            : base( node )
        {
            this.token = node.token;
            this.startIndex = node.startIndex;
            this.stopIndex = node.stopIndex;
        }

        public CommonTree( IToken t )
        {
            this.token = t;
        }

        #region Properties
        public override int CharPositionInLine
        {
            get
            {
                if ( token == null || token.CharPositionInLine == -1 )
                {
                    if ( ChildCount > 0 )
                    {
                        return GetChild( 0 ).CharPositionInLine;
                    }
                    return 0;
                }
                return token.CharPositionInLine;
            }
            set
            {
                base.CharPositionInLine = value;
            }
        }
        public override int ChildIndex
        {
            get
            {
                return childIndex;
            }
            set
            {
                childIndex = value;
            }
        }
        public override bool IsNil
        {
            get
            {
                return token == null;
            }
        }
        public override int Line
        {
            get
            {
                if ( token == null || token.Line == 0 )
                {
                    if ( ChildCount > 0 )
                    {
                        return GetChild( 0 ).Line;
                    }
                    return 0;
                }
                return token.Line;
            }
            set
            {
                base.Line = value;
            }
        }
        public override ITree Parent
        {
            get
            {
                return parent;
            }
            set
            {
                parent = (CommonTree)value;
            }
        }
        public override string Text
        {
            get
            {
                if ( token == null )
                    return null;

                return token.Text;
            }
            set
            {
            }
        }
        public virtual IToken Token
        {
            get
            {
                return token;
            }
        }
        public override int TokenStartIndex
        {
            get
            {
                if ( startIndex == -1 && token != null )
                {
                    return token.TokenIndex;
                }
                return startIndex;
            }
            set
            {
                startIndex = value;
            }
        }
        public override int TokenStopIndex
        {
            get
            {
                if ( stopIndex == -1 && token != null )
                {
                    return token.TokenIndex;
                }
                return stopIndex;
            }
            set
            {
                stopIndex = value;
            }
        }
        public override int Type
        {
            get
            {
                if ( token == null )
                    return TokenConstants.INVALID_TOKEN_TYPE;

                return token.Type;
            }
            set
            {
            }
        }
        #endregion

        public override ITree DupNode()
        {
            return new CommonTree( this );
        }

        /** <summary>
         *  For every node in this subtree, make sure it's start/stop token's
         *  are set.  Walk depth first, visit bottom up.  Only updates nodes
         *  with at least one token index &lt; 0.
         *  </summary>
         */
        public virtual void SetUnknownTokenBoundaries()
        {
            if ( children == null )
            {
                if ( startIndex < 0 || stopIndex < 0 )
                {
                    startIndex = stopIndex = token.TokenIndex;
                }
                return;
            }
            for ( int i = 0; i < children.Count; i++ )
            {
                ( (CommonTree)children[i] ).SetUnknownTokenBoundaries();
            }
            if ( startIndex >= 0 && stopIndex >= 0 )
                return; // already set
            if ( children.Count > 0 )
            {
                CommonTree firstChild = (CommonTree)children[0];
                CommonTree lastChild = (CommonTree)children[children.Count - 1];
                startIndex = firstChild.TokenStartIndex;
                stopIndex = lastChild.TokenStopIndex;
            }
        }

        public override string ToString()
        {
            if ( IsNil )
            {
                return "nil";
            }
            if ( Type == TokenConstants.INVALID_TOKEN_TYPE )
            {
                return "<errornode>";
            }
            if ( token == null )
            {
                return null;
            }
            return token.Text;
        }
    }
}
