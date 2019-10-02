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
using System.Reflection;

#if DEBUG

namespace Antlr.Runtime.JavaExtensions
{
    public static class TypeExtensions
    {
        [Obsolete]
        public static object get( this FieldInfo field, object obj )
        {
            return field.GetValue( obj );
        }

        [Obsolete]
        public static Type getComponentType( this Type type )
        {
            return type.GetElementType();
        }

        [Obsolete]
        public static ConstructorInfo getConstructor( this Type type, Type[] argumentTypes )
        {
            return type.GetConstructor( argumentTypes );
        }

        [Obsolete]
        public static FieldInfo getField( this Type type, string name )
        {
            FieldInfo field = type.GetField( name );
            if ( field == null )
                throw new TargetException();

            return field;
        }

        [Obsolete]
        public static string getName( this Type type )
        {
            return type.Name;
        }

        [Obsolete]
        public static object invoke( this MethodInfo method, object obj, params object[] parameters )
        {
            return method.Invoke( obj, parameters );
        }

        [Obsolete]
        public static bool isArray( this Type type )
        {
            return type.IsArray;
        }

        [Obsolete]
        public static bool isPrimitive( this Type type )
        {
            return type.IsPrimitive;
        }

        [Obsolete]
        public static object newInstance( this Type type )
        {
            return type.GetConstructor( new Type[0] ).Invoke( new object[0] );
        }
    }
}

#endif
