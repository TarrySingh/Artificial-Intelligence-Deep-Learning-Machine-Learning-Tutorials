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

#if DEBUG

using System;

using TextReader = System.IO.TextReader;
using TextWriter = System.IO.TextWriter;

namespace Antlr.Runtime.JavaExtensions
{
    public static class IOExtensions
    {
        [Obsolete]
        public static void close( this TextReader reader )
        {
            reader.Close();
        }

        [Obsolete]
        public static void close( this TextWriter writer )
        {
            writer.Close();
        }

        [Obsolete]
        public static void print<T>( this TextWriter writer, T value )
        {
            writer.Write( value );
        }

        [Obsolete]
        public static void println( this TextWriter writer )
        {
            writer.WriteLine();
        }

        [Obsolete]
        public static void println<T>( this TextWriter writer, T value )
        {
            writer.WriteLine( value );
        }

        [Obsolete]
        public static void write<T>( this TextWriter writer, T value )
        {
            writer.Write( value );
        }

        [Obsolete]
        public static int read( this TextReader reader, char[] buffer, int index, int count )
        {
            return reader.Read( buffer, index, count );
        }

        [Obsolete]
        public static string readLine( this TextReader reader )
        {
            return reader.ReadLine();
        }
    }
}

#endif
