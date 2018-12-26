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
    using System;
    using Antlr.Runtime.JavaExtensions;

    using IOException = System.IO.IOException;
    using ITreeAdaptor = Antlr.Runtime.Tree.ITreeAdaptor;
    using Socket = System.Net.Sockets.Socket;
    using StringBuilder = System.Text.StringBuilder;
    using TcpListener = System.Net.Sockets.TcpListener;

    /** <summary>
     *  A proxy debug event listener that forwards events over a socket to
     *  a debugger (or any other listener) using a simple text-based protocol;
     *  one event per line.  ANTLRWorks listens on server socket with a
     *  RemoteDebugEventSocketListener instance.  These two objects must therefore
     *  be kept in sync.  New events must be handled on both sides of socket.
     *  </summary>
     */
    public class DebugEventSocketProxy : BlankDebugEventListener
    {
        public const int DEFAULT_DEBUGGER_PORT = 49100;
        protected int port = DEFAULT_DEBUGGER_PORT;
        protected TcpListener serverSocket;
        protected Socket socket;
        protected string grammarFileName;
        //protected PrintWriter @out;
        //protected BufferedReader @in;

        /** <summary>Who am i debugging?</summary> */
        protected BaseRecognizer recognizer;

        /** <summary>
         *  Almost certainly the recognizer will have adaptor set, but
         *  we don't know how to cast it (Parser or TreeParser) to get
         *  the adaptor field.  Must be set with a constructor. :(
         *  </summary>
         */
        protected ITreeAdaptor adaptor;

        public DebugEventSocketProxy( BaseRecognizer recognizer, ITreeAdaptor adaptor ) :
            this( recognizer, DEFAULT_DEBUGGER_PORT, adaptor )
        {
        }

        public DebugEventSocketProxy( BaseRecognizer recognizer, int port, ITreeAdaptor adaptor )
        {
            this.grammarFileName = recognizer.GrammarFileName;
            this.adaptor = adaptor;
            this.port = port;
        }

        #region Properties
        public virtual ITreeAdaptor TreeAdaptor
        {
            get
            {
                return adaptor;
            }
            set
            {
                adaptor = value;
            }
        }
        #endregion

        public virtual void Handshake()
        {
            if ( serverSocket == null )
            {
                System.Net.IPHostEntry hostInfo = System.Net.Dns.GetHostEntry( "localhost" );
                System.Net.IPAddress ipAddress = hostInfo.AddressList[0];
                serverSocket = new TcpListener( ipAddress, port );
                socket = serverSocket.AcceptSocket();
                socket.NoDelay = true;

                System.Text.UTF8Encoding encoding = new System.Text.UTF8Encoding();
                socket.Send( encoding.GetBytes( "ANTLR " + DebugEventListenerConstants.ProtocolVersion + "\n" ) );
                socket.Send( encoding.GetBytes( "grammar \"" + grammarFileName + "\n" ) );
                Ack();

                //serverSocket = new ServerSocket( port );
                //socket = serverSocket.accept();
                //socket.setTcpNoDelay( true );
                //OutputStream os = socket.getOutputStream();
                //OutputStreamWriter osw = new OutputStreamWriter( os, "UTF8" );
                //@out = new PrintWriter( new BufferedWriter( osw ) );
                //InputStream @is = socket.getInputStream();
                //InputStreamReader isr = new InputStreamReader( @is, "UTF8" );
                //@in = new BufferedReader( isr );
                //@out.println( "ANTLR " + DebugEventListenerConstants.PROTOCOL_VERSION );
                //@out.println( "grammar \"" + grammarFileName );
                //@out.flush();
                //ack();
            }
        }

        public override void Commence()
        {
            // don't bother sending event; listener will trigger upon connection
        }

        public override void Terminate()
        {
            Transmit( "terminate" );
            //@out.close();
            try
            {
                socket.Close();
            }
            catch ( IOException ioe )
            {
                ioe.PrintStackTrace( Console.Error );
            }
        }

        protected virtual void Ack()
        {
            try
            {
                throw new NotImplementedException();
                //@in.readLine();
            }
            catch ( IOException ioe )
            {
                ioe.PrintStackTrace( Console.Error );
            }
        }

        protected virtual void Transmit( string @event )
        {
            socket.Send( new System.Text.UTF8Encoding().GetBytes( @event + "\n" ) );
            //@out.println( @event );
            //@out.flush();
            Ack();
        }

        public override void EnterRule( string grammarFileName, string ruleName )
        {
            Transmit( "enterRule\t" + grammarFileName + "\t" + ruleName );
        }

        public override void EnterAlt( int alt )
        {
            Transmit( "enterAlt\t" + alt );
        }

        public override void ExitRule( string grammarFileName, string ruleName )
        {
            Transmit( "exitRule\t" + grammarFileName + "\t" + ruleName );
        }

        public override void EnterSubRule( int decisionNumber )
        {
            Transmit( "enterSubRule\t" + decisionNumber );
        }

        public override void ExitSubRule( int decisionNumber )
        {
            Transmit( "exitSubRule\t" + decisionNumber );
        }

        public override void EnterDecision( int decisionNumber )
        {
            Transmit( "enterDecision\t" + decisionNumber );
        }

        public override void ExitDecision( int decisionNumber )
        {
            Transmit( "exitDecision\t" + decisionNumber );
        }

        public override void ConsumeToken( IToken t )
        {
            string buf = SerializeToken( t );
            Transmit( "consumeToken\t" + buf );
        }

        public override void ConsumeHiddenToken( IToken t )
        {
            string buf = SerializeToken( t );
            Transmit( "consumeHiddenToken\t" + buf );
        }

        public override void LT( int i, IToken t )
        {
            if ( t != null )
                Transmit( "LT\t" + i + "\t" + SerializeToken( t ) );
        }

        public override void Mark( int i )
        {
            Transmit( "mark\t" + i );
        }

        public override void Rewind( int i )
        {
            Transmit( "rewind\t" + i );
        }

        public override void Rewind()
        {
            Transmit( "rewind" );
        }

        public override void BeginBacktrack( int level )
        {
            Transmit( "beginBacktrack\t" + level );
        }

        public override void EndBacktrack( int level, bool successful )
        {
            Transmit( "endBacktrack\t" + level + "\t" + ( successful ? DebugEventListenerConstants.True : DebugEventListenerConstants.False ) );
        }

        public override void Location( int line, int pos )
        {
            Transmit( "location\t" + line + "\t" + pos );
        }

        public override void RecognitionException( RecognitionException e )
        {
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( "exception\t" );
            buf.Append( e.GetType().Name );
            // dump only the data common to all exceptions for now
            buf.Append( "\t" );
            buf.Append( e.index );
            buf.Append( "\t" );
            buf.Append( e.line );
            buf.Append( "\t" );
            buf.Append( e.charPositionInLine );
            Transmit( buf.ToString() );
        }

        public override void BeginResync()
        {
            Transmit( "beginResync" );
        }

        public override void EndResync()
        {
            Transmit( "endResync" );
        }

        public override void SemanticPredicate( bool result, string predicate )
        {
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( "semanticPredicate\t" );
            buf.Append( result );
            SerializeText( buf, predicate );
            Transmit( buf.ToString() );
        }

        #region AST Parsing Events

        public override void ConsumeNode( object t )
        {
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( "consumeNode" );
            SerializeNode( buf, t );
            Transmit( buf.ToString() );
        }

        public override void LT( int i, object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = adaptor.GetText( t );
            int type = adaptor.GetType( t );
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( "LN\t" ); // lookahead node; distinguish from LT in protocol
            buf.Append( i );
            SerializeNode( buf, t );
            Transmit( buf.ToString() );
        }

        protected virtual void SerializeNode( StringBuilder buf, object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = adaptor.GetText( t );
            int type = adaptor.GetType( t );
            buf.Append( "\t" );
            buf.Append( ID );
            buf.Append( "\t" );
            buf.Append( type );
            IToken token = adaptor.GetToken( t );
            int line = -1;
            int pos = -1;
            if ( token != null )
            {
                line = token.Line;
                pos = token.CharPositionInLine;
            }
            buf.Append( "\t" );
            buf.Append( line );
            buf.Append( "\t" );
            buf.Append( pos );
            int tokenIndex = adaptor.GetTokenStartIndex( t );
            buf.Append( "\t" );
            buf.Append( tokenIndex );
            SerializeText( buf, text );
        }

        #endregion


        #region AST Events

        public override void NilNode( object t )
        {
            int ID = adaptor.GetUniqueID( t );
            Transmit( "nilNode\t" + ID );
        }

        public override void ErrorNode( object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = t.ToString();
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( "errorNode\t" );
            buf.Append( ID );
            buf.Append( "\t" );
            buf.Append( TokenConstants.INVALID_TOKEN_TYPE );
            SerializeText( buf, text );
            Transmit( buf.ToString() );
        }

        public override void CreateNode( object t )
        {
            int ID = adaptor.GetUniqueID( t );
            string text = adaptor.GetText( t );
            int type = adaptor.GetType( t );
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( "createNodeFromTokenElements\t" );
            buf.Append( ID );
            buf.Append( "\t" );
            buf.Append( type );
            SerializeText( buf, text );
            Transmit( buf.ToString() );
        }

        public override void CreateNode( object node, IToken token )
        {
            int ID = adaptor.GetUniqueID( node );
            int tokenIndex = token.TokenIndex;
            Transmit( "createNode\t" + ID + "\t" + tokenIndex );
        }

        public override void BecomeRoot( object newRoot, object oldRoot )
        {
            int newRootID = adaptor.GetUniqueID( newRoot );
            int oldRootID = adaptor.GetUniqueID( oldRoot );
            Transmit( "becomeRoot\t" + newRootID + "\t" + oldRootID );
        }

        public override void AddChild( object root, object child )
        {
            int rootID = adaptor.GetUniqueID( root );
            int childID = adaptor.GetUniqueID( child );
            Transmit( "addChild\t" + rootID + "\t" + childID );
        }

        public override void SetTokenBoundaries( object t, int tokenStartIndex, int tokenStopIndex )
        {
            int ID = adaptor.GetUniqueID( t );
            Transmit( "setTokenBoundaries\t" + ID + "\t" + tokenStartIndex + "\t" + tokenStopIndex );
        }

        #endregion


        #region Support

        protected virtual string SerializeToken( IToken t )
        {
            StringBuilder buf = new StringBuilder( 50 );
            buf.Append( t.TokenIndex );
            buf.Append( '\t' );
            buf.Append( t.Type );
            buf.Append( '\t' );
            buf.Append( t.Channel );
            buf.Append( '\t' );
            buf.Append( t.Line );
            buf.Append( '\t' );
            buf.Append( t.CharPositionInLine );
            SerializeText( buf, t.Text );
            return buf.ToString();
        }

        protected virtual void SerializeText( StringBuilder buf, string text )
        {
            buf.Append( "\t\"" );
            if ( text == null )
            {
                text = "";
            }
            // escape \n and \r all text for token appears to exist on one line
            // this escape is slow but easy to understand
            text = EscapeNewlines( text );
            buf.Append( text );
        }

        protected virtual string EscapeNewlines( string txt )
        {
            txt = txt.replaceAll( "%", "%25" );   // escape all escape char ;)
            txt = txt.replaceAll( "\n", "%0A" );  // escape \n
            txt = txt.replaceAll( "\r", "%0D" );  // escape \r
            return txt;
        }

        #endregion
    }
}
