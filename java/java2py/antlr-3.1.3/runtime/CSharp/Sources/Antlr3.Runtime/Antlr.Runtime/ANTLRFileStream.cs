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
	using StreamReader	= System.IO.StreamReader;
	using FileInfo		= System.IO.FileInfo;
	using Encoding		= System.Text.Encoding;
	
	/// <summary>
	/// A character stream - an <see cref="ICharStream"/> - that loads 
	/// and caches the contents of it's underlying file fully during 
	/// object construction
	/// </summary>
	/// <remarks>
	/// This looks very much like an ANTLReaderStream or an ANTLRInputStream 
	/// but, it is a special case. Since we know the exact size of the file to 
	/// load, we can avoid lots of data copying and buffer resizing.
	/// </remarks>
	public class ANTLRFileStream : ANTLRStringStream
	{
		#region Constructors

		/// <summary>
		/// Initializes a new instance of the ANTLRFileStream class
		/// </summary>
		protected ANTLRFileStream()
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRFileStream class for the
		/// specified file name
		/// </summary>
		public ANTLRFileStream(string fileName) :
			this(fileName, Encoding.Default)
		{
		}

		/// <summary>
		/// Initializes a new instance of the ANTLRFileStream class for the
		/// specified file name and encoding
		/// </summary>
		public ANTLRFileStream(string fileName, Encoding encoding)
		{
			this.fileName = fileName;
			Load(fileName, encoding);
		}

		/// <summary>
		/// Gets the file name of this ANTLRFileStream underlying file
		/// </summary>
		override public string SourceName
		{
			get { return fileName; }
		}

		#endregion


		#region Public API

		/// <summary>
		/// Loads and buffers the specified file to be used as this 
		/// ANTLRFileStream's source
		/// </summary>
		/// <param name="fileName">File to load</param>
		/// <param name="encoding">Encoding to apply to file</param>
		public virtual void Load(string fileName, Encoding encoding)
		{
			if (fileName == null)
				return;

			StreamReader fr = null;
			try
			{
				FileInfo f = new FileInfo(fileName);
				int filelen = (int)GetFileLength(f);
				data = new char[filelen];
				if (encoding != null)
					fr = new StreamReader(fileName, encoding);
				else
					fr = new StreamReader(fileName, Encoding.Default);
				n = fr.Read((Char[])data, 0, data.Length);
			}
			finally
			{
				if (fr != null)
				{
					fr.Close();
				}
			}
		}

		#endregion


		#region Data Members

		/// <summary>Fully qualified name of the stream's underlying file</summary>
		protected string fileName;

		#endregion


		#region Private Members

		private long GetFileLength(FileInfo file)
		{
			if (file.Exists)
				return file.Length;
			else
				return 0;
		}

		#endregion
	}
}