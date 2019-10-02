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
using System.Diagnostics;
using System.Linq;

using TargetInvocationException = System.Reflection.TargetInvocationException;

namespace Antlr.Runtime.JavaExtensions
{
    public static class ExceptionExtensions
    {
#if DEBUG
        [Obsolete]
        public static string getMessage( this Exception e )
        {
            return e.Message;
        }
#endif

        public static StackFrame[] getStackTrace( this Exception e )
        {
            StackTrace trace = new StackTrace( e, true );
            StackFrame[] frames = trace.GetFrames();
            if ( frames == null )
            {
                // don't include this helper function in the trace
                frames = new StackTrace( true ).GetFrames().Skip( 1 ).ToArray();
            }
            return frames;
        }

#if DEBUG
        [Obsolete]
        public static string getMethodName( this StackFrame frame )
        {
            return frame.GetMethod().Name;
        }

        [Obsolete]
        public static string getClassName( this StackFrame frame )
        {
            return frame.GetMethod().DeclaringType.Name;
        }
#endif

        public static void PrintStackTrace( this Exception e )
        {
            e.PrintStackTrace( Console.Out );
        }
        public static void PrintStackTrace( this Exception e, System.IO.TextWriter writer )
        {
            writer.WriteLine( e.ToString() );
            foreach ( string line in e.StackTrace.Split( '\n', '\r' ) )
            {
                if ( !string.IsNullOrEmpty( line ) )
                    writer.WriteLine( "        " + line );
            }
        }

#if DEBUG
        [Obsolete]
        public static Exception getTargetException( this TargetInvocationException e )
        {
            return e.InnerException ?? e;
        }
#endif
    }
}
