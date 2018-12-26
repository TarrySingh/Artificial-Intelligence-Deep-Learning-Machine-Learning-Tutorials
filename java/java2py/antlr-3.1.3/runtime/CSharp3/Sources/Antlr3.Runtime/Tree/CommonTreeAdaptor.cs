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
     *  A TreeAdaptor that works with any Tree implementation.  It provides
     *  really just factory methods; all the work is done by BaseTreeAdaptor.
     *  If you would like to have different tokens created than ClassicToken
     *  objects, you need to override this and then set the parser tree adaptor to
     *  use your subclass.
     *  </summary>
     *
     *  <remarks>
     *  To get your parser to build nodes of a different type, override
     *  create(Token), errorNode(), and to be safe, YourTreeClass.dupNode().
     *  dupNode is called to duplicate nodes during rewrite operations.
     *  </remarks>
     */
    public class CommonTreeAdaptor : BaseTreeAdaptor
    {
        /** <summary>
         *  Duplicate a node.  This is part of the factory;
         *  override if you want another kind of node to be built.
         *  </summary>
         *
         *  <remarks>
         *  I could use reflection to prevent having to override this
         *  but reflection is slow.
         *  </remarks>
         */
        public override object DupNode( object t )
        {
            if ( t == null )
                return null;

            return ( (ITree)t ).DupNode();
        }

        public override object Create( IToken payload )
        {
            return new CommonTree( payload );
        }

        /** <summary>
         *  Tell me how to create a token for use with imaginary token nodes.
         *  For example, there is probably no input symbol associated with imaginary
         *  token DECL, but you need to create it as a payload or whatever for
         *  the DECL node as in ^(DECL type ID).
         *  </summary>
         *
         *  <remarks>
         *  If you care what the token payload objects' type is, you should
         *  override this method and any other createToken variant.
         *  </remarks>
         */
        public override IToken CreateToken( int tokenType, string text )
        {
            return new CommonToken( tokenType, text );
        }

        /** <summary>
         *  Tell me how to create a token for use with imaginary token nodes.
         *  For example, there is probably no input symbol associated with imaginary
         *  token DECL, but you need to create it as a payload or whatever for
         *  the DECL node as in ^(DECL type ID).
         *  </summary>
         *
         *  <remarks>
         *  This is a variant of createToken where the new token is derived from
         *  an actual real input token.  Typically this is for converting '{'
         *  tokens to BLOCK etc...  You'll see
         *
         *    r : lc='{' ID+ '}' -> ^(BLOCK[$lc] ID+) ;
         *
         *  If you care what the token payload objects' type is, you should
         *  override this method and any other createToken variant.
         *  </remarks>
         */
        public override IToken CreateToken( IToken fromToken )
        {
            return new CommonToken( fromToken );
        }

        /** <summary>
         *  Track start/stop token for subtree root created for a rule.
         *  Only works with Tree nodes.  For rules that match nothing,
         *  seems like this will yield start=i and stop=i-1 in a nil node.
         *  Might be useful info so I'll not force to be i..i.
         *  </summary>
         */
        public override void SetTokenBoundaries( object t, IToken startToken, IToken stopToken )
        {
            if ( t == null )
                return;

            int start = 0;
            int stop = 0;

            if ( startToken != null )
                start = startToken.TokenIndex;

            if ( stopToken != null )
                stop = stopToken.TokenIndex;

            ( (ITree)t ).TokenStartIndex = start;
            ( (ITree)t ).TokenStopIndex = stop;
        }

        public override int GetTokenStartIndex( object t )
        {
            if ( t == null )
                return -1;

            return ( (ITree)t ).TokenStartIndex;
        }

        public override int GetTokenStopIndex( object t )
        {
            if ( t == null )
                return -1;

            return ( (ITree)t ).TokenStopIndex;
        }

        public override string GetText( object t )
        {
            if ( t == null )
                return null;

            return ( (ITree)t ).Text;
        }

        public override int GetType( object t )
        {
            if ( t == null )
                return TokenConstants.INVALID_TOKEN_TYPE;

            return ( (ITree)t ).Type;
        }

        /** <summary>
         *  What is the Token associated with this node?  If
         *  you are not using CommonTree, then you must
         *  override this in your own adaptor.
         *  </summary>
         */
        public override IToken GetToken( object t )
        {
            if ( t is CommonTree )
            {
                return ( (CommonTree)t ).Token;
            }
            return null; // no idea what to do
        }

        public override object GetChild( object t, int i )
        {
            if ( t == null )
                return null;

            return ( (ITree)t ).GetChild( i );
        }

        public override int GetChildCount( object t )
        {
            if ( t == null )
                return 0;

            return ( (ITree)t ).ChildCount;
        }

        public override object GetParent( object t )
        {
            if ( t == null )
                return null;

            return ( (ITree)t ).Parent;
        }

        public override void SetParent( object t, object parent )
        {
            if ( t != null )
                ( (ITree)t ).Parent = (ITree)parent;
        }

        public override int GetChildIndex( object t )
        {
            if ( t == null )
                return 0;

            return ( (ITree)t ).ChildIndex;
        }

        public override void SetChildIndex( object t, int index )
        {
            if ( t != null )
                ( (ITree)t ).ChildIndex = index;
        }

        public override void ReplaceChildren( object parent, int startChildIndex, int stopChildIndex, object t )
        {
            if ( parent != null )
            {
                ( (ITree)parent ).ReplaceChildren( startChildIndex, stopChildIndex, t );
            }
        }
    }
}
