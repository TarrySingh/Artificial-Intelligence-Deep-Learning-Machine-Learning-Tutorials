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

    using BaseTree = Antlr.Runtime.Tree.BaseTree;
    using Console = System.Console;
    using SocketException = System.Net.Sockets.SocketException;
    using Exception = System.Exception;
    using IOException = System.IO.IOException;
    using Socket = System.Net.Sockets.Socket;
    using ITree = Antlr.Runtime.Tree.ITree;

    using TextWriter = System.IO.TextWriter;
    using TextReader = System.IO.TextReader;

    public class RemoteDebugEventSocketListener
    {
        const int MAX_EVENT_ELEMENTS = 8;
        IDebugEventListener listener;
        string machine;
        int port;
        Socket channel = null;
        TextWriter @out;
        TextReader @in;
        string @event;
        /** <summary>Version of ANTLR (dictates events)</summary> */
        public string version;
        public string grammarFileName;
        /** <summary>
         *  Track the last token index we saw during a consume.  If same, then
         *  set a flag that we have a problem.
         *  </summary>
         */
        int previousTokenIndex = -1;
        bool tokenIndexesInvalid = false;

        public class ProxyToken : IToken
        {
            int index;
            int type;
            int channel;
            int line;
            int charPos;
            string text;
            public ProxyToken( int index )
            {
                this.index = index;
            }
            public ProxyToken( int index, int type, int channel,
                              int line, int charPos, string text )
            {
                this.index = index;
                this.type = type;
                this.channel = channel;
                this.line = line;
                this.charPos = charPos;
                this.text = text;
            }

            #region IToken Members
            public string Text
            {
                get
                {
                    return text;
                }
                set
                {
                    text = value;
                }
            }

            public int Type
            {
                get
                {
                    return type;
                }
                set
                {
                    type = value;
                }
            }

            public int Line
            {
                get
                {
                    return line;
                }
                set
                {
                    line = value;
                }
            }

            public int CharPositionInLine
            {
                get
                {
                    return charPos;
                }
                set
                {
                    charPos = value;
                }
            }

            public int Channel
            {
                get
                {
                    return channel;
                }
                set
                {
                    channel = value;
                }
            }

            public int TokenIndex
            {
                get
                {
                    return index;
                }
                set
                {
                    index = value;
                }
            }

            public ICharStream InputStream
            {
                get
                {
                    return null;
                }
                set
                {
                }
            }

            #endregion

            public override string ToString()
            {
                string channelStr = "";
                if ( channel != TokenConstants.DEFAULT_CHANNEL )
                {
                    channelStr = ",channel=" + channel;
                }
                return "[" + Text + "/<" + type + ">" + channelStr + "," + line + ":" + CharPositionInLine + ",@" + index + "]";
            }
        }

        public class ProxyTree : BaseTree
        {
            public int ID;
            public int type;
            public int line = 0;
            public int charPos = -1;
            public int tokenIndex = -1;
            public string text;

            public ProxyTree( int ID, int type, int line, int charPos, int tokenIndex, string text )
            {
                this.ID = ID;
                this.type = type;
                this.line = line;
                this.charPos = charPos;
                this.tokenIndex = tokenIndex;
                this.text = text;
            }

            public ProxyTree( int ID )
            {
                this.ID = ID;
            }

            #region Properties
            public override string Text
            {
                get
                {
                    return text;
                }
                set
                {
                }
            }
            public override int TokenStartIndex
            {
                get
                {
                    return tokenIndex;
                }
                set
                {
                }
            }
            public override int TokenStopIndex
            {
                get
                {
                    return 0;
                }
                set
                {
                }
            }
            public override int Type
            {
                get
                {
                    return type;
                }
                set
                {
                }
            }
            #endregion

            public override ITree DupNode()
            {
                return null;
            }

            public override string ToString()
            {
                return "fix this";
            }
        }

        public RemoteDebugEventSocketListener( IDebugEventListener listener,
                                              string machine,
                                              int port )
        {
            this.listener = listener;
            this.machine = machine;
            this.port = port;

            if ( !OpenConnection() )
            {
                throw new SocketException();
            }
        }

        protected virtual void EventHandler()
        {
            try
            {
                Handshake();
                @event = @in.ReadLine();
                while ( @event != null )
                {
                    Dispatch( @event );
                    Ack();
                    @event = @in.ReadLine();
                }
            }
            catch ( Exception e )
            {
                Console.Error.WriteLine( e );
                e.PrintStackTrace( Console.Error );
            }
            finally
            {
                CloseConnection();
            }
        }

        protected virtual bool OpenConnection()
        {
            bool success = false;
            try
            {
                throw new System.NotImplementedException();
                //channel = new Socket( machine, port );
                //channel.setTcpNoDelay( true );
                //OutputStream os = channel.getOutputStream();
                //OutputStreamWriter osw = new OutputStreamWriter( os, "UTF8" );
                //@out = new PrintWriter( new BufferedWriter( osw ) );
                //InputStream @is = channel.getInputStream();
                //InputStreamReader isr = new InputStreamReader( @is, "UTF8" );
                //@in = new BufferedReader( isr );
                //success = true;
            }
            catch ( Exception e )
            {
                Console.Error.WriteLine( e );
            }
            return success;
        }

        protected virtual void CloseConnection()
        {
            try
            {
                @in.Close();
                @in = null;
                @out.Close();
                @out = null;
                channel.Close();
                channel = null;
            }
            catch ( Exception e )
            {
                Console.Error.WriteLine( e );
                e.PrintStackTrace( Console.Error );
            }
            finally
            {
                if ( @in != null )
                {
                    try
                    {
                        @in.Close();
                    }
                    catch ( IOException ioe )
                    {
                        Console.Error.WriteLine( ioe );
                    }
                }
                if ( @out != null )
                {
                    @out.Close();
                }
                if ( channel != null )
                {
                    try
                    {
                        channel.Close();
                    }
                    catch ( IOException ioe )
                    {
                        Console.Error.WriteLine( ioe );
                    }
                }
            }

        }

        protected virtual void Handshake()
        {
            string antlrLine = @in.ReadLine();
            string[] antlrElements = GetEventElements( antlrLine );
            version = antlrElements[1];
            string grammarLine = @in.ReadLine();
            string[] grammarElements = GetEventElements( grammarLine );
            grammarFileName = grammarElements[1];
            Ack();
            listener.Commence(); // inform listener after handshake
        }

        protected virtual void Ack()
        {
            @out.WriteLine( "ack" );
            @out.Flush();
        }

        protected virtual void Dispatch( string line )
        {
            //JSystem.@out.println( "event: " + line );
            string[] elements = GetEventElements( line );
            if ( elements == null || elements[0] == null )
            {
                Console.Error.WriteLine( "unknown debug event: " + line );
                return;
            }
            if ( elements[0].Equals( "enterRule" ) )
            {
                listener.EnterRule( elements[1], elements[2] );
            }
            else if ( elements[0].Equals( "exitRule" ) )
            {
                listener.ExitRule( elements[1], elements[2] );
            }
            else if ( elements[0].Equals( "enterAlt" ) )
            {
                listener.EnterAlt( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "enterSubRule" ) )
            {
                listener.EnterSubRule( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "exitSubRule" ) )
            {
                listener.ExitSubRule( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "enterDecision" ) )
            {
                listener.EnterDecision( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "exitDecision" ) )
            {
                listener.ExitDecision( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "location" ) )
            {
                listener.Location( int.Parse( elements[1] ),
                                  int.Parse( elements[2] ) );
            }
            else if ( elements[0].Equals( "consumeToken" ) )
            {
                ProxyToken t = DeserializeToken( elements, 1 );
                if ( t.TokenIndex == previousTokenIndex )
                {
                    tokenIndexesInvalid = true;
                }
                previousTokenIndex = t.TokenIndex;
                listener.ConsumeToken( t );
            }
            else if ( elements[0].Equals( "consumeHiddenToken" ) )
            {
                ProxyToken t = DeserializeToken( elements, 1 );
                if ( t.TokenIndex == previousTokenIndex )
                {
                    tokenIndexesInvalid = true;
                }
                previousTokenIndex = t.TokenIndex;
                listener.ConsumeHiddenToken( t );
            }
            else if ( elements[0].Equals( "LT" ) )
            {
                IToken t = DeserializeToken( elements, 2 );
                listener.LT( int.Parse( elements[1] ), t );
            }
            else if ( elements[0].Equals( "mark" ) )
            {
                listener.Mark( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "rewind" ) )
            {
                if ( elements[1] != null )
                {
                    listener.Rewind( int.Parse( elements[1] ) );
                }
                else
                {
                    listener.Rewind();
                }
            }
            else if ( elements[0].Equals( "beginBacktrack" ) )
            {
                listener.BeginBacktrack( int.Parse( elements[1] ) );
            }
            else if ( elements[0].Equals( "endBacktrack" ) )
            {
                int level = int.Parse( elements[1] );
                int successI = int.Parse( elements[2] );
                listener.EndBacktrack( level, successI == DebugEventListenerConstants.True );
            }
            else if ( elements[0].Equals( "exception" ) )
            {
#if true
                throw new System.NotImplementedException();
#else
                string excName = elements[1];
                string indexS = elements[2];
                string lineS = elements[3];
                string posS = elements[4];
                Class excClass = null;
                try
                {
                    excClass = Class.forName( excName );
                    RecognitionException e =
                        (RecognitionException)excClass.newInstance();
                    e.index = int.Parse( indexS );
                    e.line = int.Parse( lineS );
                    e.charPositionInLine = int.Parse( posS );
                    listener.recognitionException( e );
                }
                catch ( ClassNotFoundException cnfe )
                {
                    Console.Error.println( "can't find class " + cnfe );
                    cnfe.printStackTrace( Console.Error );
                }
                catch ( InstantiationException ie )
                {
                    Console.Error.println( "can't instantiate class " + ie );
                    ie.printStackTrace( Console.Error );
                }
                catch ( IllegalAccessException iae )
                {
                    Console.Error.println( "can't access class " + iae );
                    iae.printStackTrace( Console.Error );
                }
#endif
            }
            else if ( elements[0].Equals( "beginResync" ) )
            {
                listener.BeginResync();
            }
            else if ( elements[0].Equals( "endResync" ) )
            {
                listener.EndResync();
            }
            else if ( elements[0].Equals( "terminate" ) )
            {
                listener.Terminate();
            }
            else if ( elements[0].Equals( "semanticPredicate" ) )
            {
                bool result = bool.Parse( elements[1] );
                string predicateText = elements[2];
                predicateText = UnEscapeNewlines( predicateText );
                listener.SemanticPredicate( result,
                                           predicateText );
            }
            else if ( elements[0].Equals( "consumeNode" ) )
            {
                ProxyTree node = DeserializeNode( elements, 1 );
                listener.ConsumeNode( node );
            }
            else if ( elements[0].Equals( "LN" ) )
            {
                int i = int.Parse( elements[1] );
                ProxyTree node = DeserializeNode( elements, 2 );
                listener.LT( i, node );
            }
            else if ( elements[0].Equals( "createNodeFromTokenElements" ) )
            {
                int ID = int.Parse( elements[1] );
                int type = int.Parse( elements[2] );
                string text = elements[3];
                text = UnEscapeNewlines( text );
                ProxyTree node = new ProxyTree( ID, type, -1, -1, -1, text );
                listener.CreateNode( node );
            }
            else if ( elements[0].Equals( "createNode" ) )
            {
                int ID = int.Parse( elements[1] );
                int tokenIndex = int.Parse( elements[2] );
                // create dummy node/token filled with ID, tokenIndex
                ProxyTree node = new ProxyTree( ID );
                ProxyToken token = new ProxyToken( tokenIndex );
                listener.CreateNode( node, token );
            }
            else if ( elements[0].Equals( "nilNode" ) )
            {
                int ID = int.Parse( elements[1] );
                ProxyTree node = new ProxyTree( ID );
                listener.NilNode( node );
            }
            else if ( elements[0].Equals( "errorNode" ) )
            {
                // TODO: do we need a special tree here?
                int ID = int.Parse( elements[1] );
                int type = int.Parse( elements[2] );
                string text = elements[3];
                text = UnEscapeNewlines( text );
                ProxyTree node = new ProxyTree( ID, type, -1, -1, -1, text );
                listener.ErrorNode( node );
            }
            else if ( elements[0].Equals( "becomeRoot" ) )
            {
                int newRootID = int.Parse( elements[1] );
                int oldRootID = int.Parse( elements[2] );
                ProxyTree newRoot = new ProxyTree( newRootID );
                ProxyTree oldRoot = new ProxyTree( oldRootID );
                listener.BecomeRoot( newRoot, oldRoot );
            }
            else if ( elements[0].Equals( "addChild" ) )
            {
                int rootID = int.Parse( elements[1] );
                int childID = int.Parse( elements[2] );
                ProxyTree root = new ProxyTree( rootID );
                ProxyTree child = new ProxyTree( childID );
                listener.AddChild( root, child );
            }
            else if ( elements[0].Equals( "setTokenBoundaries" ) )
            {
                int ID = int.Parse( elements[1] );
                ProxyTree node = new ProxyTree( ID );
                listener.SetTokenBoundaries(
                    node,
                    int.Parse( elements[2] ),
                    int.Parse( elements[3] ) );
            }
            else
            {
                Console.Error.WriteLine( "unknown debug event: " + line );
            }
        }

        protected virtual ProxyTree DeserializeNode( string[] elements, int offset )
        {
            int ID = int.Parse( elements[offset + 0] );
            int type = int.Parse( elements[offset + 1] );
            int tokenLine = int.Parse( elements[offset + 2] );
            int charPositionInLine = int.Parse( elements[offset + 3] );
            int tokenIndex = int.Parse( elements[offset + 4] );
            string text = elements[offset + 5];
            text = UnEscapeNewlines( text );
            return new ProxyTree( ID, type, tokenLine, charPositionInLine, tokenIndex, text );
        }

        protected virtual ProxyToken DeserializeToken( string[] elements,
                                              int offset )
        {
            string indexS = elements[offset + 0];
            string typeS = elements[offset + 1];
            string channelS = elements[offset + 2];
            string lineS = elements[offset + 3];
            string posS = elements[offset + 4];
            string text = elements[offset + 5];
            text = UnEscapeNewlines( text );
            int index = int.Parse( indexS );
            ProxyToken t =
                new ProxyToken( index,
                               int.Parse( typeS ),
                               int.Parse( channelS ),
                               int.Parse( lineS ),
                               int.Parse( posS ),
                               text );
            return t;
        }

        /** <summary>Create a thread to listen to the remote running recognizer</summary> */
        public virtual void Start()
        {
            System.Threading.Thread t = new System.Threading.Thread( Run );
            t.Start();
        }

        public virtual void Run()
        {
            EventHandler();
        }

        #region Misc

        public virtual string[] GetEventElements( string @event )
        {
            if ( @event == null )
            {
                return null;
            }
            string[] elements = new string[MAX_EVENT_ELEMENTS];
            string str = null; // a string element if present (must be last)
            try
            {
                int firstQuoteIndex = @event.IndexOf( '"' );
                if ( firstQuoteIndex >= 0 )
                {
                    // treat specially; has a string argument like "a comment\n
                    // Note that the string is terminated by \n not end quote.
                    // Easier to parse that way.
                    string eventWithoutString = @event.Substring( 0, firstQuoteIndex );
                    str = @event.Substring( firstQuoteIndex + 1 );
                    @event = eventWithoutString;
                }
                StringTokenizer st = new StringTokenizer( @event, "\t", false );
                int i = 0;
                while ( st.hasMoreTokens() )
                {
                    if ( i >= MAX_EVENT_ELEMENTS )
                    {
                        // ErrorManager.internalError("event has more than "+MAX_EVENT_ELEMENTS+" args: "+event);
                        return elements;
                    }
                    elements[i] = st.nextToken();
                    i++;
                }
                if ( str != null )
                {
                    elements[i] = str;
                }
            }
            catch ( Exception e )
            {
                e.PrintStackTrace( Console.Error );
            }
            return elements;
        }

        protected virtual string UnEscapeNewlines( string txt )
        {
            // this unescape is slow but easy to understand
            txt = txt.replaceAll( "%0A", "\n" );  // unescape \n
            txt = txt.replaceAll( "%0D", "\r" );  // unescape \r
            txt = txt.replaceAll( "%25", "%" );   // undo escaped escape chars
            return txt;
        }

        public virtual bool TokenIndexesAreInvalid()
        {
            return false;
            //return tokenIndexesInvalid;
        }

        #endregion

    }
}
