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

using System;
using System.Text;

namespace Antlr.Runtime.JavaExtensions
{
    public static class StringExtensions
    {
#if DEBUG
        [Obsolete]
        public static char charAt( this string str, int index )
        {
            return str[index];
        }

        [Obsolete]
        public static bool endsWith( this string str, string value )
        {
            return str.EndsWith( value );
        }

        [Obsolete]
        public static int indexOf( this string str, char value )
        {
            return str.IndexOf( value );
        }

        [Obsolete]
        public static int indexOf( this string str, char value, int startIndex )
        {
            return str.IndexOf( value, startIndex );
        }

        [Obsolete]
        public static int indexOf( this string str, string value )
        {
            return str.IndexOf( value );
        }

        [Obsolete]
        public static int indexOf( this string str, string value, int startIndex )
        {
            return str.IndexOf( value, startIndex );
        }

        [Obsolete]
        public static int lastIndexOf( this string str, char value )
        {
            return str.LastIndexOf( value );
        }

        [Obsolete]
        public static int lastIndexOf( this string str, string value )
        {
            return str.LastIndexOf( value );
        }

        [Obsolete]
        public static int length( this string str )
        {
            return str.Length;
        }
#endif

        public static string replace( this string str, char oldValue, char newValue )
        {
            int index = str.IndexOf( oldValue );
            if ( index == -1 )
                return str;

            System.Text.StringBuilder builder = new StringBuilder( str );
            builder[index] = newValue;
            return builder.ToString();
        }

        public static string replaceAll( this string str, string regex, string newValue )
        {
            return System.Text.RegularExpressions.Regex.Replace( str, regex, newValue );
        }

        public static string replaceFirst( this string str, string regex, string replacement )
        {
            return System.Text.RegularExpressions.Regex.Replace( str, regex, replacement );
        }

#if DEBUG
        [Obsolete]
        public static bool startsWith( this string str, string value )
        {
            return str.StartsWith( value );
        }
#endif

        [Obsolete]
        public static string substring( this string str, int startOffset )
        {
            return str.Substring( startOffset );
        }

        public static string substring( this string str, int startOffset, int endOffset )
        {
            return str.Substring( startOffset, endOffset - startOffset );
        }

#if DEBUG
        [Obsolete]
        public static char[] toCharArray( this string str )
        {
            return str.ToCharArray();
        }

        [Obsolete]
        public static string toUpperCase( this string str )
        {
            return str.ToUpperInvariant();
        }

        [Obsolete]
        public static string trim( this string str )
        {
            return str.Trim();
        }
#endif
    }
}
