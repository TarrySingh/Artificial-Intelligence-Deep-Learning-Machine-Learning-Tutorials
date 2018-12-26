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
using System.Linq;

namespace Antlr.Runtime.JavaExtensions
{
    public class StringTokenizer
    {
        string[] _tokens;
        int _current;

        public StringTokenizer( string str, string separator )
            : this( str, separator, false )
        {
        }
        public StringTokenizer( string str, string separator, bool returnDelims )
        {
            _tokens = str.Split( separator.ToCharArray(), StringSplitOptions.None );
            if ( returnDelims )
            {
                char[] delims = separator.ToCharArray();
                _tokens = _tokens.SelectMany( ( token, i ) =>
                {
                    if ( i == _tokens.Length - 1 )
                    {
                        if ( delims.Contains( str[str.Length - 1] ) )
                            return new string[0];
                        else
                            return new string[] { token };
                    }
                    else if ( i == 0 )
                    {
                        if ( delims.Contains( str[0] ) )
                            return new string[] { str[0].ToString() };
                        else
                            return new string[] { token };
                    }
                    else
                    {
                        return new string[] { token, str[_tokens.Take( i + 1 ).Select( t => t.Length + 1 ).Sum() - 1].ToString() };
                    }
                } ).ToArray();
            }
        }

        public bool hasMoreTokens()
        {
            return _current < _tokens.Length;
        }

        public string nextToken()
        {
            return _tokens[_current++];
        }
    }
}
