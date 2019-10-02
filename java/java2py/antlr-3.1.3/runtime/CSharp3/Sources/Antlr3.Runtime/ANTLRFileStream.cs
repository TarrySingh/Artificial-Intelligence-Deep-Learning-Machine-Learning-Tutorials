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
    using Encoding = System.Text.Encoding;

    /** <summary>
     *  This is a char buffer stream that is loaded from a file
     *  all at once when you construct the object.  This looks very
     *  much like an ANTLReader or ANTLRInputStream, but it's a special case
     *  since we know the exact size of the object to load.  We can avoid lots
     *  of data copying. 
     *  </summary>
     */
    [System.Serializable]
    public class ANTLRFileStream : ANTLRStringStream
    {
        protected string fileName;

        public ANTLRFileStream( string fileName ) 
            : this( fileName, null )
        {
        }

        public ANTLRFileStream( string fileName, Encoding encoding )
        {
            this.fileName = fileName;
            Load( fileName, encoding );
        }

        public virtual void Load( string fileName, Encoding encoding )
        {
            if ( fileName == null )
            {
                return;
            }

            data = System.IO.File.ReadAllText( fileName, encoding ).ToCharArray();
            n = data.Length;
        }

        public override string SourceName
        {
            get
            {
                return fileName;
            }
        }
    }
}
