/*
 [The "BSD licence"]
 Copyright (c) 2005-2008 Terence Parr
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. The name of the author may not be used to endorse or promote products
    derived from this software without specific prior written permission.

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
package org.antlr.runtime;

import java.io.*;

/** Vacuum all input from a Reader and then treat it like a StringStream.
 *  Manage the buffer manually to avoid unnecessary data copying.
 *
 *  If you need encoding, use ANTLRInputStream.
 */
public class ANTLRReaderStream extends ANTLRStringStream {
	public static final int READ_BUFFER_SIZE = 1024;
	public static final int INITIAL_BUFFER_SIZE = 1024;

	public ANTLRReaderStream() {
	}

	public ANTLRReaderStream(Reader r) throws IOException {
		this(r, INITIAL_BUFFER_SIZE, READ_BUFFER_SIZE);
	}

	public ANTLRReaderStream(Reader r, int size) throws IOException {
		this(r, size, READ_BUFFER_SIZE);
	}

	public ANTLRReaderStream(Reader r, int size, int readChunkSize) throws IOException {
		load(r, size, readChunkSize);
	}

	public void load(Reader r, int size, int readChunkSize)
		throws IOException
	{
		if ( r==null ) {
			return;
		}
		if ( size<=0 ) {
			size = INITIAL_BUFFER_SIZE;
		}
		if ( readChunkSize<=0 ) {
			readChunkSize = READ_BUFFER_SIZE;
		}
		// System.out.println("load "+size+" in chunks of "+readChunkSize);
		try {
			// alloc initial buffer size.
			data = new char[size];
			// read all the data in chunks of readChunkSize
			int numRead=0;
			int p = 0;
			do {
				if ( p+readChunkSize > data.length ) { // overflow?
					// System.out.println("### overflow p="+p+", data.length="+data.length);
					char[] newdata = new char[data.length*2]; // resize
					System.arraycopy(data, 0, newdata, 0, data.length);
					data = newdata;
				}
				numRead = r.read(data, p, readChunkSize);
				// System.out.println("read "+numRead+" chars; p was "+p+" is now "+(p+numRead));
				p += numRead;
			} while (numRead!=-1); // while not EOF
			// set the actual size of the data available;
			// EOF subtracted one above in p+=numRead; add one back
			super.n = p+1;
			//System.out.println("n="+n);
		}
		finally {
			r.close();
		}
	}
}
