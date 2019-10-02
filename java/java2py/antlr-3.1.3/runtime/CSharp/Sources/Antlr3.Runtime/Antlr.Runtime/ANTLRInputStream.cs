/*
[The "BSD licence"]
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
	using Encoding		= System.Text.Encoding;
	using Stream		= System.IO.Stream;
	using StreamReader	= System.IO.StreamReader;

	/// <summary>
	/// A character stream - an <see cref="ICharStream"/> - that loads 
	/// and caches the contents of it's underlying 
	/// <see cref="System.IO.Stream"/> fully during object construction
	/// </summary>
	/// <remarks>
	/// Useful for reading from stdin and, for specifying file encodings etc...
	/// </remarks>
	public class ANTLRInputStream : ANTLRReaderStream
	{
		#region Constructors

		/// <summary>
		/// Initializes a new instance of the ANTLRInputStream class
		/// </summary>
		protected ANTLRInputStream()
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRInputStream class for the
		/// specified stream
		/// </summary>
		public ANTLRInputStream(Stream istream)
			: this(istream, null)
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRInputStream class for the
		/// specified stream and encoding
		/// </summary>
		public ANTLRInputStream(Stream istream, Encoding encoding)
			: this(istream, INITIAL_BUFFER_SIZE, encoding)
		{

		}

		/// <summary>
		/// Initializes a new instance of the ANTLRInputStream class for the
		/// specified stream and initial data buffer size
		/// </summary>
		public ANTLRInputStream(Stream istream, int size)
			: this(istream, size, null)
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRInputStream class for the
		/// specified stream, encoding and initial data buffer size
		/// </summary>
		public ANTLRInputStream(Stream istream, int size, Encoding encoding)
			: this(istream, size, READ_BUFFER_SIZE, encoding)
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRInputStream class for the
		/// specified stream, encoding, initial data buffer size and, using 
		/// a read buffer of the specified size
		/// </summary>
		public ANTLRInputStream(Stream istream, int size, int readBufferSize, Encoding encoding)
		{
			StreamReader reader;
			if (encoding != null)
			{
				reader = new StreamReader(istream, encoding);
			}
			else
			{
				reader = new StreamReader(istream);
			}
			Load(reader, size, readBufferSize);
		}

		#endregion

	}
}