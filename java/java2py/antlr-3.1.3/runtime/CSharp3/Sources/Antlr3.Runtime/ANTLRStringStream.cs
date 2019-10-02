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

namespace Antlr.Runtime
{
    using System.Collections.Generic;

    /** <summary>
     *  A pretty quick CharStream that pulls all data from an array
     *  directly.  Every method call counts in the lexer.  Java's
     *  strings aren't very good so I'm avoiding.
     *  </summary>
     */
    [System.Serializable]
    public class ANTLRStringStream : ICharStream
    {
        /** <summary>The data being scanned</summary> */
        protected char[] data;

        /** <summary>How many characters are actually in the buffer</summary> */
        protected int n;

        /** <summary>0..n-1 index into string of next char</summary> */
        protected int p = 0;

        /** <summary>line number 1..n within the input</summary> */
        protected int line = 1;

        /** <summary>The index of the character relative to the beginning of the line 0..n-1</summary> */
        protected int charPositionInLine = 0;

        /** <summary>tracks how deep mark() calls are nested</summary> */
        protected int markDepth = 0;

        /** <summary>
         *  A list of CharStreamState objects that tracks the stream state
         *  values line, charPositionInLine, and p that can change as you
         *  move through the input stream.  Indexed from 1..markDepth.
         *  A null is kept @ index 0.  Create upon first call to mark().
         *  </summary>
         */
        protected IList<CharStreamState> markers;

        /** <summary>Track the last mark() call result value for use in rewind().</summary> */
        protected int lastMarker;

        /** <summary>What is name or source of this char stream?</summary> */
        public string name;

        public ANTLRStringStream()
        {
        }

        /** <summary>Copy data in string to a local char array</summary> */
        public ANTLRStringStream( string input )
            : this( input, null )
        {
        }

        public ANTLRStringStream( string input, string sourceName )
            : this( input.ToCharArray(), input.Length, sourceName )
        {
        }

        /** <summary>This is the preferred constructor as no data is copied</summary> */
        public ANTLRStringStream( char[] data, int numberOfActualCharsInArray )
            : this( data, numberOfActualCharsInArray, null )
        {
        }

        public ANTLRStringStream( char[] data, int numberOfActualCharsInArray, string sourceName )
            : this()
        {
            this.data = data;
            this.n = numberOfActualCharsInArray;
            this.name = sourceName;
        }

        /** <summary>
         *  Return the current input symbol index 0..n where n indicates the
         *  last symbol has been read.  The index is the index of char to
         *  be returned from LA(1).
         *  </summary>
         */
        public virtual int Index
        {
            get
            {
                return p;
            }
        }
        public virtual int Line
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
        public virtual int CharPositionInLine
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

        /** <summary>
         *  Reset the stream so that it's in the same state it was
         *  when the object was created *except* the data array is not
         *  touched.
         *  </summary>
         */
        public virtual void Reset()
        {
            p = 0;
            line = 1;
            charPositionInLine = 0;
            markDepth = 0;
        }

        public virtual void Consume()
        {
            //System.out.println("prev p="+p+", c="+(char)data[p]);
            if ( p < n )
            {
                charPositionInLine++;
                if ( data[p] == '\n' )
                {
                    /*
                    System.out.println("newline char found on line: "+line+
                                       "@ pos="+charPositionInLine);
                    */
                    line++;
                    charPositionInLine = 0;
                }
                p++;
                //System.out.println("p moves to "+p+" (c='"+(char)data[p]+"')");
            }
        }

        public virtual int LA( int i )
        {
            if ( i == 0 )
            {
                return 0; // undefined
            }
            if ( i < 0 )
            {
                i++; // e.g., translate LA(-1) to use offset i=0; then data[p+0-1]
                if ( ( p + i - 1 ) < 0 )
                {
                    return CharStreamConstants.EOF; // invalid; no char before first char
                }
            }

            if ( ( p + i - 1 ) >= n )
            {
                //System.out.println("char LA("+i+")=EOF; p="+p);
                return CharStreamConstants.EOF;
            }
            //System.out.println("char LA("+i+")="+(char)data[p+i-1]+"; p="+p);
            //System.out.println("LA("+i+"); p="+p+" n="+n+" data.length="+data.length);
            return data[p + i - 1];
        }

        public virtual int LT( int i )
        {
            return LA( i );
        }

        public virtual int Size()
        {
            return n;
        }

        public virtual int Mark()
        {
            if ( markers == null )
            {
                markers = new List<CharStreamState>();
                markers.Add( null ); // depth 0 means no backtracking, leave blank
            }
            markDepth++;
            CharStreamState state = null;
            if ( markDepth >= markers.Count )
            {
                state = new CharStreamState();
                markers.Add( state );
            }
            else
            {
                state = markers[markDepth];
            }
            state.p = p;
            state.line = line;
            state.charPositionInLine = charPositionInLine;
            lastMarker = markDepth;
            return markDepth;
        }

        public virtual void Rewind( int m )
        {
            CharStreamState state = markers[m];
            // restore stream state
            Seek( state.p );
            line = state.line;
            charPositionInLine = state.charPositionInLine;
            Release( m );
        }

        public virtual void Rewind()
        {
            Rewind( lastMarker );
        }

        public virtual void Release( int marker )
        {
            // unwind any other markers made after m and release m
            markDepth = marker;
            // release this marker
            markDepth--;
        }

        /** <summary>
         *  consume() ahead until p==index; can't just set p=index as we must
         *  update line and charPositionInLine.
         *  </summary>
         */
        public virtual void Seek( int index )
        {
            if ( index <= p )
            {
                p = index; // just jump; don't update stream state (line, ...)
                return;
            }
            // seek forward, consume until p hits index
            while ( p < index )
            {
                Consume();
            }
        }

        public virtual string substring( int start, int stop )
        {
            return new string( data, start, stop - start + 1 );
        }

        public virtual string SourceName
        {
            get
            {
                return name;
            }
        }
    }
}
