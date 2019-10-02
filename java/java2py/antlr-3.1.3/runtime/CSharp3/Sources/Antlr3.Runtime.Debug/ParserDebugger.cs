namespace Antlr.Runtime.Debug
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    using IOException = System.IO.IOException;
    using Antlr.Runtime.Tree;

    public class ParserDebugger
    {
        IDebugEventListener dbg;

        public ParserDebugger( Parser parser )
            : this( parser, DebugEventSocketProxy.DEFAULT_DEBUGGER_PORT )
        {
        }
        public ParserDebugger( Parser parser, int port )
        {
            DebugEventSocketProxy proxy = new DebugEventSocketProxy( parser, port, null );
            DebugListener = proxy;
            parser.TokenStream = new DebugTokenStream( parser.TokenStream, proxy );
            try
            {
                proxy.handshake();
            }
            catch ( IOException e )
            {
                reportError( ioe );
            }
            ITreeAdaptor adap = new CommonTreeAdaptor();
            TreeAdaptor = adap;
            proxy.TreeAdaptor = adap;
        }
        public ParserDebugger( Parser parser, IDebugEventListener dbg )
        {
            ITreeAdaptor adap = new CommonTreeAdaptor();
            TreeAdaptor = adap;
        }

        protected virtual bool EvalPredicate( bool result, string predicate )
        {
            dbg.SemanticPredicate( result, predicate );
            return result;
        }

    }
}
