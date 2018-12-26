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

namespace Antlr.Runtime.JavaExtensions
{
    public static class ObjectExtensions
    {
#if DEBUG
        [Obsolete]
        public static bool booleanValue( this bool value )
        {
            return value;
        }

        [Obsolete]
        public static Type getClass( this object o )
        {
            return o.GetType();
        }
#endif

        public static int ShiftPrimeXOR( int a, int b )
        {
            int hash = 23;
            hash = ( ( hash << 5 ) * 37 ) ^ a;
            hash = ( ( hash << 5 ) * 37 ) ^ b;
            return hash;
        }

        public static int ShiftPrimeXOR( int a, int b, int c )
        {
            int hash = 23;
            hash = ( ( hash << 5 ) * 37 ) ^ a;
            hash = ( ( hash << 5 ) * 37 ) ^ b;
            hash = ( ( hash << 5 ) * 37 ) ^ c;
            return hash;
        }

        public static int ShiftPrimeXOR( int a, int b, int c, int d )
        {
            int hash = 23;
            hash = ( ( hash << 5 ) * 37 ) ^ a;
            hash = ( ( hash << 5 ) * 37 ) ^ b;
            hash = ( ( hash << 5 ) * 37 ) ^ c;
            hash = ( ( hash << 5 ) * 37 ) ^ d;
            return hash;
        }

        public static int ShiftPrimeXOR( params int[] a )
        {
            int hash = 23;
            foreach ( int i in a )
                hash = ( ( hash << 5 ) * 37 ) ^ i;
            return hash;
        }

        public static int ShiftPrimeAdd( int a, int b )
        {
            int hash = 23;
            hash = ( ( hash << 5 ) * 37 ) + a;
            hash = ( ( hash << 5 ) * 37 ) + b;
            return hash;
        }

        public static int ShiftPrimeAdd( int a, int b, int c )
        {
            int hash = 23;
            hash = ( ( hash << 5 ) * 37 ) + a;
            hash = ( ( hash << 5 ) * 37 ) + b;
            hash = ( ( hash << 5 ) * 37 ) + c;
            return hash;
        }

        public static int ShiftPrimeAdd( int a, int b, int c, int d )
        {
            int hash = 23;
            hash = ( ( hash << 5 ) * 37 ) + a;
            hash = ( ( hash << 5 ) * 37 ) + b;
            hash = ( ( hash << 5 ) * 37 ) + c;
            hash = ( ( hash << 5 ) * 37 ) + d;
            return hash;
        }

        public static int ShiftPrimeAdd( params int[] a )
        {
            int hash = 23;
            foreach ( int i in a )
                hash = ( ( hash << 5 ) * 37 ) + i;
            return hash;
        }
    }
}
