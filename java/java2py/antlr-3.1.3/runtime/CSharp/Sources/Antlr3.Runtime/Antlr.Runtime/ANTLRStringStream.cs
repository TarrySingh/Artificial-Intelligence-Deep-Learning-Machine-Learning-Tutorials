/*
[The "BSD licence"]
Copyright (c) 2007-2008 Johannes Luber
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


namespace Antlr.Runtime
{
    using System;
    using IList                 = System.Collections.IList;
    using ArrayList             = System.Collections.ArrayList;

	/// <summary>
	/// A pretty quick <see cref="ICharStream"/> that uses a character array 
	/// directly as it's underlying source.
	/// </summary>
    public class ANTLRStringStream : ICharStream
    {
        #region Constructors

		/// <summary>
		/// Initializes a new instance of the ANTLRStringStream class
		/// </summary>
		protected ANTLRStringStream()
        {
        }

		/// <summary>
		/// Initializes a new instance of the ANTLRStringStream class for the
		/// specified string. This copies data from the string to a local 
		/// character array
		/// </summary>
        public ANTLRStringStream(string input)
        {
            this.data = input.ToCharArray();
			this.n = input.Length;
        }

		/// <summary>
		/// Initializes a new instance of the ANTLRStringStream class for the
		/// specified character array. This is the preferred constructor as 
		/// no data is copied
		/// </summary>
		public ANTLRStringStream(char[] data, int numberOfActualCharsInArray)
        {
            this.data = data;
			this.n = numberOfActualCharsInArray;
		}

        #endregion

        #region Public API

		/// <summary>
		/// Current line position in stream.
		/// </summary>
		virtual public int Line
        {
            get { return line; }
            set { this.line = value; }

        }

		/// <summary>
		/// Current character position on the current line stream 
		/// (i.e. columnn position)
		/// </summary>
		virtual public int CharPositionInLine
        {
            get { return charPositionInLine; }
            set { this.charPositionInLine = value; }

        }

        /// <summary>
        /// Resets the stream so that it is in the same state it was
        /// when the object was created *except* the data array is not
        /// touched.
        /// </summary>
        public virtual void Reset()
        {
            p = 0;
            line = 1;
            charPositionInLine = 0;
            markDepth = 0;
        }

		/// <summary>
		/// Advances the read position of the stream. Updates line and column state
		/// </summary>
		public virtual void Consume()
        {
            if (p < n)
            {
                charPositionInLine++;
                if (data[p] == '\n')
                {
                    line++;
                    charPositionInLine = 0;
                }
                p++;
            }
        }

		/// <summary>
		/// Return lookahead characters at the specified offset from the current read position.
		/// The lookahead offset can be negative.
		/// </summary>
		public virtual int LA(int i)
        {
			if (i == 0)
			{
				return 0; // undefined
			}
			if (i < 0)
			{
				i++; // e.g., translate LA(-1) to use offset i=0; then data[p+0-1]
				if ( (p + i - 1) < 0 ) 
				{
					return (int)CharStreamConstants.EOF; // invalid; no char before first char
				}
			}
			
			if ((p + i - 1) >= n)
            {
                return (int)CharStreamConstants.EOF;
            }
            return data[p + i - 1];
        }

        public virtual int LT(int i)
        {
            return LA(i);
        }

        /// <summary>
        /// Return the current input symbol index 0..n where n indicates the
		/// last symbol has been read. The index is the index of char to
		/// be returned from LA(1).
        /// </summary>
        public virtual int Index()
        {
            return p;
        }

		/// <summary>
		/// Returns the size of the stream
		/// </summary>
		[Obsolete("Please use property Count instead.")]
		public virtual int Size()
        {
			return Count;
        }

		/// <summary>
		/// Returns the size of the stream
		/// </summary>
		public virtual int Count
        {
			get { return n; }
        }

        public virtual int Mark()
        {
			if (markers == null)
			{
				markers = new ArrayList();
				markers.Add(null); // depth 0 means no backtracking, leave blank
			}
            markDepth++;
            CharStreamState state = null;
            if (markDepth >= markers.Count)
            {
                state = new CharStreamState();
                markers.Add(state);
            }
            else
            {
                state = (CharStreamState)markers[markDepth];
            }
            state.p = p;
            state.line = line;
            state.charPositionInLine = charPositionInLine;
			lastMarker = markDepth;
            return markDepth;
        }

        public virtual void Rewind(int m)
        {
            CharStreamState state = (CharStreamState)markers[m];
            // restore stream state
            Seek(state.p);
            line = state.line;
            charPositionInLine = state.charPositionInLine;
			Release(m);
		}

		public virtual void Rewind() 
		{
			Rewind(lastMarker);
		}

		public virtual void Release(int marker)
        {
            // unwind any other markers made after m and release m
            markDepth = marker;
            // release this marker
            markDepth--;
        }

        /// <summary>Seeks to the specified position.</summary>
        /// <remarks>
        /// Consume ahead until p==index; can't just set p=index as we must
        /// update line and charPositionInLine.
        /// </remarks>
        public virtual void Seek(int index)
        {
            if (index <= p)
            {
                p = index; // just jump; don't update stream state (line, ...)
                return;
            }
            // seek forward, consume until p hits index
            while (p < index)
            {
                Consume();
            }
        }

        public virtual string Substring(int start, int stop)
        {
            return new string(data, start, stop - start + 1);
        }
		
		public virtual string SourceName {
			get { return name; }
			set { name = value; }
		}

        #endregion


        #region Data Members

        /// <summary>The data for the stream</summary>
        protected internal char[] data;

		/// <summary>How many characters are actually in the buffer?</summary>
		protected int n;


        /// <summary>Index in our array for the next char (0..n-1)</summary>
        protected internal int p = 0;

        /// <summary>Current line number within the input (1..n )</summary>
        protected internal int line = 1;

        /// <summary>
        /// The index of the character relative to the beginning of the 
        /// line (0..n-1)
        /// </summary>
        protected internal int charPositionInLine = 0;

        /// <summary>
        /// Tracks the depth of nested <see cref="IIntStream.Mark"/> calls
        /// </summary>
        protected internal int markDepth = 0;

        /// <summary>
        /// A list of CharStreamState objects that tracks the stream state
        /// (i.e. line, charPositionInLine, and p) that can change as you
        /// move through the input stream.  Indexed from 1..markDepth.
        /// A null is kept @ index 0.  Create upon first call to Mark().
        /// </summary>
        protected internal IList markers;

		/// <summary>
		/// Track the last Mark() call result value for use in Rewind().
		/// </summary>
		protected int lastMarker;

		/// <summary>
		/// What is name or source of this char stream?
		/// </summary>
		protected string name;

		#endregion

    }
}