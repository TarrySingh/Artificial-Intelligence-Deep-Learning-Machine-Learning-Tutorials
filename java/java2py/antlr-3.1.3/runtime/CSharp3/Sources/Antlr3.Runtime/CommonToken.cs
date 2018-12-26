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
    using NonSerialized = System.NonSerializedAttribute;
    using Regex = System.Text.RegularExpressions.Regex;
    using Serializable = System.SerializableAttribute;

    [Serializable]
    public class CommonToken : IToken
    {
        int type;
        int line;
        int charPositionInLine = -1; // set to invalid position
        int channel = TokenConstants.DEFAULT_CHANNEL;
        [NonSerialized]
        ICharStream input;

        /** <summary>
         *  We need to be able to change the text once in a while.  If
         *  this is non-null, then getText should return this.  Note that
         *  start/stop are not affected by changing this.
         *  </summary>
          */
        string text;

        /** <summary>What token number is this from 0..n-1 tokens; &lt; 0 implies invalid index</summary> */
        int index = -1;

        /** <summary>The char position into the input buffer where this token starts</summary> */
        int start;

        /** <summary>The char position into the input buffer where this token stops</summary> */
        int stop;

        public CommonToken( int type )
        {
            this.type = type;
        }

        public CommonToken( ICharStream input, int type, int channel, int start, int stop )
        {
            this.input = input;
            this.type = type;
            this.channel = channel;
            this.start = start;
            this.stop = stop;
        }

        public CommonToken( int type, string text )
        {
            this.type = type;
            this.channel = TokenConstants.DEFAULT_CHANNEL;
            this.text = text;
        }

        public CommonToken( IToken oldToken )
        {
            text = oldToken.Text;
            type = oldToken.Type;
            line = oldToken.Line;
            index = oldToken.TokenIndex;
            charPositionInLine = oldToken.CharPositionInLine;
            channel = oldToken.Channel;
            if ( oldToken is CommonToken )
            {
                start = ( (CommonToken)oldToken ).start;
                stop = ( (CommonToken)oldToken ).stop;
            }
        }

        #region IToken Members
        public string Text
        {
            get
            {
                if ( text != null )
                {
                    return text;
                }
                if ( input == null )
                {
                    return null;
                }
                text = input.substring( start, stop );
                return text;
            }
            set
            {
                /** Override the text for this token.  getText() will return this text
                 *  rather than pulling from the buffer.  Note that this does not mean
                 *  that start/stop indexes are not valid.  It means that that input
                 *  was converted to a new string in the token object.
                 */
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
                return input;
            }
            set
            {
                input = value;
            }
        }

        #endregion

        public int StartIndex
        {
            get
            {
                return start;
            }
            set
            {
                start = value;
            }
        }

        public int StopIndex
        {
            get
            {
                return stop;
            }
            set
            {
                stop = value;
            }
        }

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
            return "[@" + TokenIndex + "," + start + ":" + stop + "='" + txt + "',<" + type + ">" + channelStr + "," + line + ":" + CharPositionInLine + "]";
        }

        [System.Runtime.Serialization.OnSerializing]
        internal void OnSerializing( System.Runtime.Serialization.StreamingContext context )
        {
            if ( text == null )
                text = Text;
        }
    }
}
