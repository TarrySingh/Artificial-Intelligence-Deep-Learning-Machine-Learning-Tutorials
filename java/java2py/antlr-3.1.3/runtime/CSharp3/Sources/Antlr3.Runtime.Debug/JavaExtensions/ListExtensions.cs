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
using System.Linq;

using ICollection = System.Collections.ICollection;
using IEnumerable = System.Collections.IEnumerable;
using IList = System.Collections.IList;

namespace Antlr.Runtime.JavaExtensions
{
    public static class ListExtensions
    {
#if DEBUG
        [Obsolete]
        public static bool add( this IList list, object value )
        {
            int count = list.Count;
            list.Add( value );
            return list.Count == count + 1;
        }

        [Obsolete]
        public static void add<T>( this ICollection<T> list, T value )
        {
            list.Add( value );
        }

        [Obsolete]
        public static void add<T>( this List<T> list, T value )
        {
            list.Add( value );
        }

        [Obsolete]
        public static void add( this IList list, int index, object value )
        {
            list.Insert( index, value );
        }

        [Obsolete]
        public static void addAll( this List<object> list, IEnumerable items )
        {
            list.AddRange( items.Cast<object>() );
        }
#endif

        public static void addAll( this IList list, IEnumerable items )
        {
            foreach ( object item in items )
                list.Add( item );
        }

        public static void addAll<T>( this ICollection<T> list, IEnumerable<T> items )
        {
            foreach ( T item in items )
                list.Add( item );
        }

#if DEBUG
        [Obsolete]
        public static void addElement( this List<object> list, object value )
        {
            list.Add( value );
        }

        [Obsolete]
        public static void clear( this IList list )
        {
            list.Clear();
        }

        [Obsolete]
        public static bool contains( this IList list, object value )
        {
            return list.Contains( value );
        }

        [Obsolete]
        public static bool contains<T>( this ICollection<T> list, T value )
        {
            return list.Contains( value );
        }

        [Obsolete]
        public static T elementAt<T>( this IList<T> list, int index )
        {
            return list[index];
        }

        [Obsolete]
        public static object get( this IList list, int index )
        {
            return list[index];
        }

        [Obsolete]
        public static T get<T>( this IList<T> list, int index )
        {
            return list[index];
        }

        // disambiguate
        [Obsolete]
        public static T get<T>( this List<T> list, int index )
        {
            return list[index];
        }

        [Obsolete]
        public static object remove( this IList list, int index )
        {
            object o = list[index];
            list.RemoveAt( index );
            return o;
        }

        [Obsolete]
        public static void remove<T>( this IList<T> list, T item )
        {
            list.Remove( item );
        }

        [Obsolete]
        public static void set( this IList list, int index, object value )
        {
            list[index] = value;
        }

        [Obsolete]
        public static void set<T>( this IList<T> list, int index, T value )
        {
            list[index] = value;
        }

        [Obsolete]
        public static void set<T>( this List<T> list, int index, T value )
        {
            list[index] = value;
        }
#endif

        public static void setSize<T>( this List<T> list, int size )
        {
            if ( list.Count < size )
            {
                list.AddRange( Enumerable.Repeat( default( T ), size - list.Count ) );
            }
            else if ( list.Count > size )
            {
                T[] items = list.Take( size ).ToArray();
                list.Clear();
                list.AddRange( items );
            }
        }

#if DEBUG
        [Obsolete]
        public static int size( this ICollection collection )
        {
            return collection.Count;
        }

        [Obsolete]
        public static int size<T>( this ICollection<T> collection )
        {
            return collection.Count;
        }

        [Obsolete]
        public static int size<T>( this List<T> list )
        {
            return list.Count;
        }
#endif

        public static IList subList( this IList list, int fromIndex, int toIndex )
        {
            return new SubList( list, fromIndex, toIndex );
            //return
            //    list
            //    .Cast<object>()
            //    .Skip( fromIndex )
            //    .Take( toIndex - fromIndex + 1 )
            //    .ToList();
        }

        public static IList<T> subList<T>( this IList<T> list, int fromIndex, int toIndex )
        {
            return new SubList<T>( list, fromIndex, toIndex );
            //return
            //    list
            //    .Skip( fromIndex )
            //    .Take( toIndex - fromIndex + 1 )
            //    .ToList();
        }

        public static IList<T> subList<T>( this List<T> list, int fromIndex, int toIndex )
        {
            return new SubList<T>( list, fromIndex, toIndex );
            //return
            //    list
            //    .Skip( fromIndex )
            //    .Take( toIndex - fromIndex + 1 )
            //    .ToList();
        }
    }
}
