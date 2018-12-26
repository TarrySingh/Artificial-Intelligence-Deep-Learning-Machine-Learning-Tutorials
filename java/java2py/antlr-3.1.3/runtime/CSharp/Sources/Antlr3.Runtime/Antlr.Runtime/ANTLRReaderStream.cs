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
	using TextReader = System.IO.TextReader;

	/// <summary>
	/// An ANTLRStringStream that caches all the input from a TextReader. It 
	/// behaves just like a plain ANTLRStringStream
	/// </summary>
	/// <remarks>
	/// Manages the buffer manually to avoid unnecessary data copying.
	/// If you need encoding, use ANTLRInputStream.
	/// </remarks>
	public class ANTLRReaderStream : ANTLRStringStream
	{
		/// <summary>Default size (in characters) of the buffer used for IO reads</summary>
		public static readonly int READ_BUFFER_SIZE = 1024;

		/// <summary>Initial size (in characters) of the data cache</summary>
		public static readonly int INITIAL_BUFFER_SIZE = 1024;

		#region Constructors

		/// <summary>
		/// Initializes a new instance of the ANTLRReaderStream class
		/// </summary>
		protected ANTLRReaderStream()
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRReaderStream class for the
		/// specified TextReader
		/// </summary>
		public ANTLRReaderStream(TextReader reader)
			: this(reader, INITIAL_BUFFER_SIZE, READ_BUFFER_SIZE)
		{

		}

		/// <summary>
		/// Initializes a new instance of the ANTLRReaderStream class for the
		/// specified TextReader and initial data buffer size
		/// </summary>
		public ANTLRReaderStream(TextReader reader, int size)
			: this(reader, size, READ_BUFFER_SIZE)
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRReaderStream class for the
		/// specified TextReader, initial data buffer size and, using 
		/// a read buffer of the specified size
		/// </summary>
		public ANTLRReaderStream(TextReader reader, int size, int readChunkSize)
		{
			Load(reader, size, readChunkSize);
		}

		#endregion


		#region Public API

		/// <summary>
		/// Loads and buffers the contents of the specified reader to be 
		/// used as this ANTLRReaderStream's source
		/// </summary>
		public virtual void Load(TextReader reader, int size, int readChunkSize)
		{
			if (reader == null)
			{
				return;
			}
			if (size <= 0)
			{
				size = INITIAL_BUFFER_SIZE;
			}
			if (readChunkSize <= 0)
			{
				readChunkSize = READ_BUFFER_SIZE;
			}

			try
			{
				// alloc initial buffer size.
				data = new char[size];
				// read all the data in chunks of readChunkSize
				int numRead = 0;
				int p = 0;
				do
				{
					if (p + readChunkSize > data.Length)
					{ // overflow?
						char[] newdata = new char[data.Length * 2]; // resize
						Array.Copy(data, 0, newdata, 0, data.Length);
						data = newdata;
					}
					numRead = reader.Read(data, p, readChunkSize);
					p += numRead;
				} while (numRead != 0); // while not EOF
				// set the actual size of the data available;
				// EOF subtracted one above in p+=numRead; add one back
				//n = p + 1;
				n = p;
			}
			finally
			{
				reader.Close();
			}
		}

		#endregion
	}
}
