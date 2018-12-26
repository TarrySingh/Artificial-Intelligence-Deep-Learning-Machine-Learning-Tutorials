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
    using TextReader = System.IO.TextReader;

    /** <summary>
     *  Vacuum all input from a Reader and then treat it like a StringStream.
     *  Manage the buffer manually to avoid unnecessary data copying.
     *  </summary>
     *
     *  <remarks>
     *  If you need encoding, use ANTLRInputStream.
     *  </remarks>
     */
    [System.Serializable]
    public class ANTLRReaderStream : ANTLRStringStream
    {
        public const int READ_BUFFER_SIZE = 1024;
        public const int INITIAL_BUFFER_SIZE = 1024;

        public ANTLRReaderStream()
        {
        }

        public ANTLRReaderStream( TextReader r )
            : this( r, INITIAL_BUFFER_SIZE, READ_BUFFER_SIZE )
        {
        }

        public ANTLRReaderStream( TextReader r, int size )
            : this( r, size, READ_BUFFER_SIZE )
        {
        }

        public ANTLRReaderStream( TextReader r, int size, int readChunkSize )
        {
            Load( r, size, readChunkSize );
        }

        public virtual void Load( TextReader r, int size, int readChunkSize )
        {
            if ( r == null )
            {
                return;
            }
            if ( size <= 0 )
            {
                size = INITIAL_BUFFER_SIZE;
            }
            if ( readChunkSize <= 0 )
            {
                readChunkSize = READ_BUFFER_SIZE;
            }
            // System.out.println("load "+size+" in chunks of "+readChunkSize);
            try
            {
                data = r.ReadToEnd().ToCharArray();
                base.n = data.Length;
            }
            finally
            {
                r.Close();
            }
        }
    }
}
