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

namespace Antlr.Runtime
{
    using Regex = System.Text.RegularExpressions.Regex;

    /** <summary>
     *  A Token object like we'd use in ANTLR 2.x; has an actual string created
     *  and associated with this object.  These objects are needed for imaginary
     *  tree nodes that have payload objects.  We need to create a Token object
     *  that has a string; the tree node will point at this token.  CommonToken
     *  has indexes into a char stream and hence cannot be used to introduce
     *  new strings.
     *  </summary>
     */
    [System.Serializable]
    public class ClassicToken : IToken
    {
        protected string text;
        protected int type;
        protected int line;
        protected int charPositionInLine;
        protected int channel = TokenConstants.DEFAULT_CHANNEL;

        /** <summary>What token number is this from 0..n-1 tokens</summary> */
        protected int index;

        public ClassicToken( int type )
        {
            this.type = type;
        }

        public ClassicToken( IToken oldToken )
        {
            text = oldToken.Text;
            type = oldToken.Type;
            line = oldToken.Line;
            charPositionInLine = oldToken.CharPositionInLine;
            channel = oldToken.Channel;
        }

        public ClassicToken( int type, string text )
        {
            this.type = type;
            this.text = text;
        }

        public ClassicToken( int type, string text, int channel )
        {
            this.type = type;
            this.text = text;
            this.channel = channel;
        }

        #region IToken Members
        public string Text
        {
            get
            {
                return text;
            }
            set
            {
                text = value;
            }
        }

        public int Type
        {
            get
            {
                return type;
            }
            set
            {
                type = value;
            }
        }

        public int Line
        {
            get
            {
                return line;
            }
            set
            {
                line = value;
            }
        }

        public int CharPositionInLine
        {
            get
            {
                return charPositionInLine;
            }
            set
            {
                charPositionInLine = value;
            }
        }

        public int Channel
        {
            get
            {
                return channel;
            }
            set
            {
                channel = value;
            }
        }

        public int TokenIndex
        {
            get
            {
                return index;
            }
            set
            {
                index = value;
            }
        }

        public ICharStream InputStream
        {
            get
            {
                return null;
            }
            set
            {
            }
        }

        #endregion

        public override string ToString()
        {
            string channelStr = "";
            if ( channel > 0 )
            {
                channelStr = ",channel=" + channel;
            }
            string txt = Text;
            if ( txt != null )
            {
                txt = Regex.Replace( txt, "\n", "\\\\n" );
                txt = Regex.Replace( txt, "\r", "\\\\r" );
                txt = Regex.Replace( txt, "\t", "\\\\t" );
            }
            else
            {
                txt = "<no text>";
            }
            return "[@" + TokenIndex + ",'" + txt + "',<" + type + ">" + channelStr + "," + line + ":" + CharPositionInLine + "]";
        }
    }
}
