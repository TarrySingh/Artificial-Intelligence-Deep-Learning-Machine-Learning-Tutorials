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

namespace Antlr.Runtime.Debug
{
    using Antlr.Runtime.JavaExtensions;

    using Console = System.Console;
    using IOException = System.IO.IOException;

    public class DebugParser : Parser
    {
        /** <summary>Who to notify when events in the parser occur.</summary> */
        public IDebugEventListener dbg = null;

        /** <summary>
         *  Used to differentiate between fixed lookahead and cyclic DFA decisions
         *  while profiling.
         *  </summary>
         */
        public bool isCyclicDecision = false;

        /** <summary>
         *  Create a normal parser except wrap the token stream in a debug
         *  proxy that fires consume events.
         *  </summary>
         */
        public DebugParser( ITokenStream input, IDebugEventListener dbg, RecognizerSharedState state )
            : base( input is DebugTokenStream ? input : new DebugTokenStream( input, dbg ), state )
        {
            DebugListener = dbg;
        }

        public DebugParser( ITokenStream input, RecognizerSharedState state )
            : base( input is DebugTokenStream ? input : new DebugTokenStream( input, null ), state )
        {
        }

        public DebugParser( ITokenStream input, IDebugEventListener dbg )
            : this( input is DebugTokenStream ? input : new DebugTokenStream( input, dbg ), dbg, null )
        {
        }

        /** <summary>
         *  Provide a new debug event listener for this parser.  Notify the
         *  input stream too that it should send events to this listener.
         *  </summary>
         */
        public virtual IDebugEventListener DebugListener
        {
            get
            {
                return dbg;
            }
            set
            {
                DebugTokenStream debugTokenStream = input as DebugTokenStream;
                if ( debugTokenStream != null )
                    debugTokenStream.DebugListener = value;

                dbg = value;
            }
        }

        public virtual void ReportError( IOException e )
        {
            Console.Error.WriteLine( e );
            e.PrintStackTrace( Console.Error );
        }

        public override void BeginResync()
        {
            dbg.BeginResync();
            base.BeginResync();
        }

        public override void EndResync()
        {
            dbg.EndResync();
            base.EndResync();
        }

        public virtual void BeginBacktrack( int level )
        {
            dbg.BeginBacktrack( level );
        }

        public virtual void EndBacktrack( int level, bool successful )
        {
            dbg.EndBacktrack( level, successful );
        }

        public override void ReportError( RecognitionException e )
        {
            dbg.RecognitionException( e );
        }
    }
}
