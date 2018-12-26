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

namespace Antlr.Runtime
{
    using Antlr.Runtime.Tree;

    using Exception = System.Exception;
    using NonSerialized = System.NonSerializedAttribute;

    /** <summary>The root of the ANTLR exception hierarchy.</summary>
     *
     *  <remarks>
     *  To avoid English-only error messages and to generally make things
     *  as flexible as possible, these exceptions are not created with strings,
     *  but rather the information necessary to generate an error.  Then
     *  the various reporting methods in Parser and Lexer can be overridden
     *  to generate a localized error message.  For example, MismatchedToken
     *  exceptions are built with the expected token type.
     *  So, don't expect getMessage() to return anything.
     *
     *  Note that as of Java 1.4, you can access the stack trace, which means
     *  that you can compute the complete trace of rules from the start symbol.
     *  This gives you considerable context information with which to generate
     *  useful error messages.
     *
     *  ANTLR generates code that throws exceptions upon recognition error and
     *  also generates code to catch these exceptions in each rule.  If you
     *  want to quit upon first error, you can turn off the automatic error
     *  handling mechanism using rulecatch action, but you still need to
     *  override methods mismatch and recoverFromMismatchSet.
     *
     *  In general, the recognition exceptions can track where in a grammar a
     *  problem occurred and/or what was the expected input.  While the parser
     *  knows its state (such as current input symbol and line info) that
     *  state can change before the exception is reported so current token index
     *  is computed and stored at exception time.  From this info, you can
     *  perhaps print an entire line of input not just a single token, for example.
     *  Better to just say the recognizer had a problem and then let the parser
     *  figure out a fancy report.
     *  </remarks>
     */
    [System.Serializable]
    public class RecognitionException : Exception
    {
        /** <summary>What input stream did the error occur in?</summary> */
        [NonSerialized]
        public IIntStream input;

        /** <summary>What is index of token/char were we looking at when the error occurred?</summary> */
        public int index;

        /** <summary>
         *  The current Token when an error occurred.  Since not all streams
         *  can retrieve the ith Token, we have to track the Token object.
         *  For parsers.  Even when it's a tree parser, token might be set.
         *  </summary>
         */
        public IToken token;

        /** <summary>
         *  If this is a tree parser exception, node is set to the node with
         *  the problem.
         *  </summary>
         */
        public object node;

        /** <summary>The current char when an error occurred. For lexers.</summary> */
        public int c;

        /** <summary>
         *  Track the line at which the error occurred in case this is
         *  generated from a lexer.  We need to track this since the
         *  unexpected char doesn't carry the line info.
         *  </summary>
         */
        public int line;

        public int charPositionInLine;

        /** <summary>
         *  If you are parsing a tree node stream, you will encounter som
         *  imaginary nodes w/o line/col info.  We now search backwards looking
         *  for most recent token with line/col info, but notify getErrorHeader()
         *  that info is approximate.
         *  </summary>
         */
        public bool approximateLineInfo;

        /** <summary>Used for remote debugger deserialization</summary> */
        public RecognitionException()
        {
        }

        public RecognitionException( IIntStream input )
        {
            this.input = input;
            this.index = input.Index;
            if ( input is ITokenStream )
            {
                this.token = ( (ITokenStream)input ).LT( 1 );
                this.line = token.Line;
                this.charPositionInLine = token.CharPositionInLine;
            }
            if ( input is ITreeNodeStream )
            {
                ExtractInformationFromTreeNodeStream( input );
            }
            else if ( input is ICharStream )
            {
                this.c = input.LA( 1 );
                this.line = ( (ICharStream)input ).Line;
                this.charPositionInLine = ( (ICharStream)input ).CharPositionInLine;
            }
            else
            {
                this.c = input.LA( 1 );
            }
        }

        protected virtual void ExtractInformationFromTreeNodeStream( IIntStream input )
        {
            ITreeNodeStream nodes = (ITreeNodeStream)input;
            this.node = nodes.LT( 1 );
            ITreeAdaptor adaptor = nodes.TreeAdaptor;
            IToken payload = adaptor.GetToken( node );
            if ( payload != null )
            {
                this.token = payload;
                if ( payload.Line <= 0 )
                {
                    // imaginary node; no line/pos info; scan backwards
                    int i = -1;
                    object priorNode = nodes.LT( i );
                    while ( priorNode != null )
                    {
                        IToken priorPayload = adaptor.GetToken( priorNode );
                        if ( priorPayload != null && priorPayload.Line > 0 )
                        {
                            // we found the most recent real line / pos info
                            this.line = priorPayload.Line;
                            this.charPositionInLine = priorPayload.CharPositionInLine;
                            this.approximateLineInfo = true;
                            break;
                        }
                        --i;
                        priorNode = nodes.LT( i );
                    }
                }
                else
                { // node created from real token
                    this.line = payload.Line;
                    this.charPositionInLine = payload.CharPositionInLine;
                }
            }
            else if ( this.node is Tree.ITree )
            {
                this.line = ( (Tree.ITree)this.node ).Line;
                this.charPositionInLine = ( (Tree.ITree)this.node ).CharPositionInLine;
                if ( this.node is CommonTree )
                {
                    this.token = ( (CommonTree)this.node ).token;
                }
            }
            else
            {
                int type = adaptor.GetType( this.node );
                string text = adaptor.GetText( this.node );
                this.token = new CommonToken( type, text );
            }
        }

        /** <summary>Return the token type or char of the unexpected input element</summary> */
        public virtual int UnexpectedType
        {
            get
            {
                if ( input is ITokenStream )
                {
                    return token.Type;
                }

                ITreeNodeStream treeNodeStream = input as ITreeNodeStream;
                if ( treeNodeStream != null )
                {
                    ITreeAdaptor adaptor = treeNodeStream.TreeAdaptor;
                    return adaptor.GetType( node );
                }

                return c;
            }
        }
    }
}
