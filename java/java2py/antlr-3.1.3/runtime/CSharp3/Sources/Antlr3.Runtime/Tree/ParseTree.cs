/*
 * [The "BSD licence"]
 * Copyright (c) 2005-2008 Terence Parr
 * All rights reserved.
 *
 * Conversion to C#:
 * Copyright (c) 2008 Sam Harwell, Pixel Mine, Inc.
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

    using StringBuilder = System.Text.StringBuilder;

    /** <summary>
     *  A record of the rules used to match a token sequence.  The tokens
     *  end up as the leaves of this tree and rule nodes are the interior nodes.
     *  This really adds no functionality, it is just an alias for CommonTree
     *  that is more meaningful (specific) and holds a String to display for a node.
     *  </summary>
     */
    [System.Serializable]
    public class ParseTree : BaseTree
    {
        public object payload;
        public List<IToken> hiddenTokens;

        public ParseTree( object label )
        {
            this.payload = label;
        }

        #region Properties
        public override string Text
        {
            get
            {
                return ToString();
            }
            set
            {
            }
        }
        public override int TokenStartIndex
        {
            get
            {
                return 0;
            }
            set
            {
            }
        }
        public override int TokenStopIndex
        {
            get
            {
                return 0;
            }
            set
            {
            }
        }
        public override int Type
        {
            get
            {
                return 0;
            }
            set
            {
            }
        }
        #endregion

        public override ITree DupNode()
        {
            return null;
        }

        public override string ToString()
        {
            if ( payload is IToken )
            {
                IToken t = (IToken)payload;
                if ( t.Type == TokenConstants.EOF )
                {
                    return "<EOF>";
                }
                return t.Text;
            }
            return payload.ToString();
        }

        /** <summary>
         *  Emit a token and all hidden nodes before.  EOF node holds all
         *  hidden tokens after last real token.
         *  </summary>
         */
        public virtual string ToStringWithHiddenTokens()
        {
            StringBuilder buf = new StringBuilder();
            if ( hiddenTokens != null )
            {
                for ( int i = 0; i < hiddenTokens.Count; i++ )
                {
                    IToken hidden = (IToken)hiddenTokens[i];
                    buf.Append( hidden.Text );
                }
            }
            string nodeText = this.ToString();
            if ( !nodeText.Equals( "<EOF>" ) )
                buf.Append( nodeText );
            return buf.ToString();
        }

        /** <summary>
         *  Print out the leaves of this tree, which means printing original
         *  input back out.
         *  </summary>
         */
        public virtual string ToInputString()
        {
            StringBuilder buf = new StringBuilder();
            _ToStringLeaves( buf );
            return buf.ToString();
        }

        public virtual void _ToStringLeaves( StringBuilder buf )
        {
            if ( payload is IToken )
            { // leaf node token?
                buf.Append( this.ToStringWithHiddenTokens() );
                return;
            }
            for ( int i = 0; children != null && i < children.Count; i++ )
            {
                ParseTree t = (ParseTree)children[i];
                t._ToStringLeaves( buf );
            }
        }
    }
}
