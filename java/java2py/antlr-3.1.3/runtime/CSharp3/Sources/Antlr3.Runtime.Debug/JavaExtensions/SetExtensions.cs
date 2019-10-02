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
using System.Collections.Generic;
#if DEBUG
using System.Linq;
#endif

namespace Antlr.Runtime.JavaExtensions
{
    public static class SetExtensions
    {
#if DEBUG
        [Obsolete]
        public static bool add<T>( this HashSet<T> set, T item )
        {
            return set.Add( item );
        }
#endif

        public static void addAll<T>( this HashSet<T> set, IEnumerable<T> items )
        {
            foreach ( T item in items )
                set.Add( item );
        }

#if DEBUG
        [Obsolete]
        public static void clear<T>( this HashSet<T> set )
        {
            set.Clear();
        }

        [Obsolete]
        public static bool contains<T>( this HashSet<T> set, T value )
        {
            return set.Contains( value );
        }

        [Obsolete]
        public static bool remove<T>( this HashSet<T> set, T item )
        {
            return set.Remove( item );
        }

        [Obsolete]
        public static int size<T>( this HashSet<T> set )
        {
            return set.Count;
        }

        [Obsolete]
        public static T[] toArray<T>( this HashSet<T> set )
        {
            return set.ToArray();
        }
#endif
    }
}
