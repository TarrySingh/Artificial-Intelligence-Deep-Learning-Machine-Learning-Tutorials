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
    using System.Collections.Generic;

    using ArgumentException = System.ArgumentException;
    using Exception = System.Exception;
    using StringBuilder = System.Text.StringBuilder;
    using Type = System.Type;

    /** Useful for dumping out the input stream after doing some
     *  augmentation or other manipulations.
     *
     *  You can insert stuff, replace, and delete chunks.  Note that the
     *  operations are done lazily--only if you convert the buffer to a
     *  String.  This is very efficient because you are not moving data around
     *  all the time.  As the buffer of tokens is converted to strings, the
     *  toString() method(s) check to see if there is an operation at the
     *  current index.  If so, the operation is done and then normal String
     *  rendering continues on the buffer.  This is like having multiple Turing
     *  machine instruction streams (programs) operating on a single input tape. :)
     *
     *  Since the operations are done lazily at toString-time, operations do not
     *  screw up the token index values.  That is, an insert operation at token
     *  index i does not change the index values for tokens i+1..n-1.
     *
     *  Because operations never actually alter the buffer, you may always get
     *  the original token stream back without undoing anything.  Since
     *  the instructions are queued up, you can easily simulate transactions and
     *  roll back any changes if there is an error just by removing instructions.
     *  For example,
     *
     *   CharStream input = new ANTLRFileStream("input");
     *   TLexer lex = new TLexer(input);
     *   TokenRewriteStream tokens = new TokenRewriteStream(lex);
     *   T parser = new T(tokens);
     *   parser.startRule();
     *
     * 	 Then in the rules, you can execute
     *      Token t,u;
     *      ...
     *      input.insertAfter(t, "text to put after t");}
     * 		input.insertAfter(u, "text after u");}
     * 		System.out.println(tokens.toString());
     *
     *  Actually, you have to cast the 'input' to a TokenRewriteStream. :(
     *
     *  You can also have multiple "instruction streams" and get multiple
     *  rewrites from a single pass over the input.  Just name the instruction
     *  streams and use that name again when printing the buffer.  This could be
     *  useful for generating a C file and also its header file--all from the
     *  same buffer:
     *
     *      tokens.insertAfter("pass1", t, "text to put after t");}
     * 		tokens.insertAfter("pass2", u, "text after u");}
     * 		System.out.println(tokens.toString("pass1"));
     * 		System.out.println(tokens.toString("pass2"));
     *
     *  If you don't use named rewrite streams, a "default" stream is used as
     *  the first example shows.
     */
    [System.Serializable]
    public class TokenRewriteStream : CommonTokenStream
    {
        public const string DEFAULT_PROGRAM_NAME = "default";
        public const int PROGRAM_INIT_SIZE = 100;
        public const int MIN_TOKEN_INDEX = 0;

        // Define the rewrite operation hierarchy

        protected class RewriteOperation
        {
            /** <summary>What index into rewrites List are we?</summary> */
            public int instructionIndex;
            /** <summary>Token buffer index.</summary> */
            public int index;
            public object text;
            // outer
            protected TokenRewriteStream stream;

            protected RewriteOperation( TokenRewriteStream stream, int index, object text )
            {
                this.index = index;
                this.text = text;
                this.stream = stream;
            }
            /** <summary>
             *  Execute the rewrite operation by possibly adding to the buffer.
             *  Return the index of the next token to operate on.
             *  </summary>
             */
            public virtual int Execute( StringBuilder buf )
            {
                return index;
            }
            public override string ToString()
            {
                string opName = this.GetType().Name;
                int index = opName.IndexOf( '$' );
                opName = opName.Substring( index + 1 );
                return "<" + opName + "@" + this.index + ":\"" + text + "\">";
            }
        }

        class InsertBeforeOp : RewriteOperation
        {
            public InsertBeforeOp( TokenRewriteStream stream, int index, object text ) :
                base( stream, index, text )
            {
            }
            public override int Execute( StringBuilder buf )
            {
                buf.Append( text );
                buf.Append( ( (IToken)stream.tokens[index] ).Text );
                return index + 1;
            }
        }

        /** <summary>
         *  I'm going to try replacing range from x..y with (y-x)+1 ReplaceOp
         *  instructions.
         *  </summary>
         */
        class ReplaceOp : RewriteOperation
        {
            public int lastIndex;
            public ReplaceOp( TokenRewriteStream stream, int from, int to, object text )
                : base( stream, from, text )
            {
                lastIndex = to;
            }
            public override int Execute( StringBuilder buf )
            {
                if ( text != null )
                {
                    buf.Append( text );
                }
                return lastIndex + 1;
            }
            public override string ToString()
            {
                return "<ReplaceOp@" + index + ".." + lastIndex + ":\"" + text + "\">";
            }
        }

        class DeleteOp : ReplaceOp
        {
            public DeleteOp( TokenRewriteStream stream, int from, int to ) :
                base( stream, from, to, null )
            {
            }
            public override string ToString()
            {
                return "<DeleteOp@" + index + ".." + lastIndex + ">";
            }
        }

        /** <summary>
         *  You may have multiple, named streams of rewrite operations.
         *  I'm calling these things "programs."
         *  Maps String (name) -> rewrite (List)
         *  </summary>
         */
        protected IDictionary<string, IList<RewriteOperation>> programs = null;

        /** <summary>Map String (program name) -> Integer index</summary> */
        protected IDictionary<string, int> lastRewriteTokenIndexes = null;

        public TokenRewriteStream()
        {
            Init();
        }

        protected void Init()
        {
            programs = new Dictionary<string, IList<RewriteOperation>>();
            programs[DEFAULT_PROGRAM_NAME] = new List<RewriteOperation>( PROGRAM_INIT_SIZE );
            lastRewriteTokenIndexes = new Dictionary<string, int>();
        }

        public TokenRewriteStream( ITokenSource tokenSource )
            : base( tokenSource )
        {
            Init();
        }

        public TokenRewriteStream( ITokenSource tokenSource, int channel )
            : base( tokenSource, channel )
        {
            Init();
        }

        public virtual void Rollback( int instructionIndex )
        {
            Rollback( DEFAULT_PROGRAM_NAME, instructionIndex );
        }

        /** <summary>
         *  Rollback the instruction stream for a program so that
         *  the indicated instruction (via instructionIndex) is no
         *  longer in the stream.  UNTESTED!
         *  </summary>
         */
        public virtual void Rollback( string programName, int instructionIndex )
        {
            IList<RewriteOperation> @is;
            if ( programs.TryGetValue( programName, out @is ) && @is != null )
            {
                List<RewriteOperation> sublist = new List<RewriteOperation>();
                for ( int i = MIN_TOKEN_INDEX; i <= instructionIndex; i++ )
                    sublist.Add( @is[i] );

                programs[programName] = sublist;
            }
        }

        public virtual void DeleteProgram()
        {
            DeleteProgram( DEFAULT_PROGRAM_NAME );
        }

        /** <summary>Reset the program so that no instructions exist</summary> */
        public virtual void DeleteProgram( string programName )
        {
            Rollback( programName, MIN_TOKEN_INDEX );
        }

        public virtual void InsertAfter( IToken t, object text )
        {
            InsertAfter( DEFAULT_PROGRAM_NAME, t, text );
        }

        public virtual void InsertAfter( int index, object text )
        {
            InsertAfter( DEFAULT_PROGRAM_NAME, index, text );
        }

        public virtual void InsertAfter( string programName, IToken t, object text )
        {
            InsertAfter( programName, t.TokenIndex, text );
        }

        public virtual void InsertAfter( string programName, int index, object text )
        {
            // to insert after, just insert before next index (even if past end)
            InsertBefore( programName, index + 1, text );
            //addToSortedRewriteList(programName, new InsertAfterOp(index,text));
        }

        public virtual void InsertBefore( IToken t, object text )
        {
            InsertBefore( DEFAULT_PROGRAM_NAME, t, text );
        }

        public virtual void InsertBefore( int index, object text )
        {
            InsertBefore( DEFAULT_PROGRAM_NAME, index, text );
        }

        public virtual void InsertBefore( string programName, IToken t, object text )
        {
            InsertBefore( programName, t.TokenIndex, text );
        }

        public virtual void InsertBefore( string programName, int index, object text )
        {
            //addToSortedRewriteList(programName, new InsertBeforeOp(index,text));
            RewriteOperation op = new InsertBeforeOp( this, index, text );
            IList<RewriteOperation> rewrites = GetProgram( programName );
            op.instructionIndex = rewrites.Count;
            rewrites.Add( op );
        }

        public virtual void Replace( int index, object text )
        {
            Replace( DEFAULT_PROGRAM_NAME, index, index, text );
        }

        public virtual void Replace( int from, int to, object text )
        {
            Replace( DEFAULT_PROGRAM_NAME, from, to, text );
        }

        public virtual void Replace( IToken indexT, object text )
        {
            Replace( DEFAULT_PROGRAM_NAME, indexT, indexT, text );
        }

        public virtual void Replace( IToken from, IToken to, object text )
        {
            Replace( DEFAULT_PROGRAM_NAME, from, to, text );
        }

        public virtual void Replace( string programName, int from, int to, object text )
        {
            if ( from > to || from < 0 || to < 0 || to >= tokens.Count )
            {
                throw new ArgumentException( "replace: range invalid: " + from + ".." + to + "(size=" + tokens.Count + ")" );
            }
            RewriteOperation op = new ReplaceOp( this, from, to, text );
            IList<RewriteOperation> rewrites = GetProgram( programName );
            op.instructionIndex = rewrites.Count;
            rewrites.Add( op );
        }

        public virtual void Replace( string programName, IToken from, IToken to, object text )
        {
            Replace( programName,
                    from.TokenIndex,
                    to.TokenIndex,
                    text );
        }

        public virtual void Delete( int index )
        {
            Delete( DEFAULT_PROGRAM_NAME, index, index );
        }

        public virtual void Delete( int from, int to )
        {
            Delete( DEFAULT_PROGRAM_NAME, from, to );
        }

        public virtual void Delete( IToken indexT )
        {
            Delete( DEFAULT_PROGRAM_NAME, indexT, indexT );
        }

        public virtual void Delete( IToken from, IToken to )
        {
            Delete( DEFAULT_PROGRAM_NAME, from, to );
        }

        public virtual void Delete( string programName, int from, int to )
        {
            Replace( programName, from, to, null );
        }

        public virtual void Delete( string programName, IToken from, IToken to )
        {
            Replace( programName, from, to, null );
        }

        public virtual int GetLastRewriteTokenIndex()
        {
            return GetLastRewriteTokenIndex( DEFAULT_PROGRAM_NAME );
        }

        protected virtual int GetLastRewriteTokenIndex( string programName )
        {
            int value;
            if ( lastRewriteTokenIndexes.TryGetValue( programName, out value ) )
                return value;

            return -1;
        }

        protected virtual void SetLastRewriteTokenIndex( string programName, int i )
        {
            lastRewriteTokenIndexes[programName] = i;
        }

        protected virtual IList<RewriteOperation> GetProgram( string name )
        {
            IList<RewriteOperation> @is;
            if ( !programs.TryGetValue( name, out @is ) || @is == null )
            {
                @is = InitializeProgram( name );
            }
            return @is;
        }

        private IList<RewriteOperation> InitializeProgram( string name )
        {
            IList<RewriteOperation> @is = new List<RewriteOperation>( PROGRAM_INIT_SIZE );
            programs[name] = @is;
            return @is;
        }

        public virtual string ToOriginalString()
        {
            return ToOriginalString( MIN_TOKEN_INDEX, Size() - 1 );
        }

        public virtual string ToOriginalString( int start, int end )
        {
            StringBuilder buf = new StringBuilder();
            for ( int i = start; i >= MIN_TOKEN_INDEX && i <= end && i < tokens.Count; i++ )
            {
                buf.Append( Get( i ).Text );
            }
            return buf.ToString();
        }

        public override string ToString()
        {
            return ToString( MIN_TOKEN_INDEX, Size() - 1 );
        }

        public virtual string ToString( string programName )
        {
            return ToString( programName, MIN_TOKEN_INDEX, Size() - 1 );
        }

        public override string ToString( int start, int end )
        {
            return ToString( DEFAULT_PROGRAM_NAME, start, end );
        }

        public virtual string ToString( string programName, int start, int end )
        {
            IList<RewriteOperation> rewrites;
            if ( !programs.TryGetValue( programName, out rewrites ) )
                rewrites = null;

            // ensure start/end are in range
            if ( end > tokens.Count - 1 )
                end = tokens.Count - 1;
            if ( start < 0 )
                start = 0;

            if ( rewrites == null || rewrites.Count == 0 )
            {
                return ToOriginalString( start, end ); // no instructions to execute
            }
            StringBuilder buf = new StringBuilder();

            // First, optimize instruction stream
            IDictionary<int, RewriteOperation> indexToOp = ReduceToSingleOperationPerIndex( rewrites );

            // Walk buffer, executing instructions and emitting tokens
            int i = start;
            while ( i <= end && i < tokens.Count )
            {
                RewriteOperation op;
                bool exists = indexToOp.TryGetValue( i, out op );

                if ( exists )
                {
                    // remove so any left have index size-1
                    indexToOp.Remove( i );
                }

                if ( !exists || op == null )
                {
                    IToken t = tokens[i];
                    // no operation at that index, just dump token
                    buf.Append( t.Text );
                    i++; // move to next token
                }
                else
                {
                    i = op.Execute( buf ); // execute operation and skip
                }
            }

            // include stuff after end if it's last index in buffer
            // So, if they did an insertAfter(lastValidIndex, "foo"), include
            // foo if end==lastValidIndex.
            if ( end == tokens.Count - 1 )
            {
                // Scan any remaining operations after last token
                // should be included (they will be inserts).
                foreach ( RewriteOperation op in indexToOp.Values )
                {
                    if ( op.index >= tokens.Count - 1 )
                        buf.Append( op.text );
                }
            }
            return buf.ToString();
        }

        /** We need to combine operations and report invalid operations (like
         *  overlapping replaces that are not completed nested).  Inserts to
         *  same index need to be combined etc...   Here are the cases:
         *
         *  I.i.u I.j.v								leave alone, nonoverlapping
         *  I.i.u I.i.v								combine: Iivu
         *
         *  R.i-j.u R.x-y.v	| i-j in x-y			delete first R
         *  R.i-j.u R.i-j.v							delete first R
         *  R.i-j.u R.x-y.v	| x-y in i-j			ERROR
         *  R.i-j.u R.x-y.v	| boundaries overlap	ERROR
         *
         *  I.i.u R.x-y.v | i in x-y				delete I
         *  I.i.u R.x-y.v | i not in x-y			leave alone, nonoverlapping
         *  R.x-y.v I.i.u | i in x-y				ERROR
         *  R.x-y.v I.x.u 							R.x-y.uv (combine, delete I)
         *  R.x-y.v I.i.u | i not in x-y			leave alone, nonoverlapping
         *
         *  I.i.u = insert u before op @ index i
         *  R.x-y.u = replace x-y indexed tokens with u
         *
         *  First we need to examine replaces.  For any replace op:
         *
         * 		1. wipe out any insertions before op within that range.
         *		2. Drop any replace op before that is contained completely within
         *         that range.
         *		3. Throw exception upon boundary overlap with any previous replace.
         *
         *  Then we can deal with inserts:
         *
         * 		1. for any inserts to same index, combine even if not adjacent.
         * 		2. for any prior replace with same left boundary, combine this
         *         insert with replace and delete this replace.
         * 		3. throw exception if index in same range as previous replace
         *
         *  Don't actually delete; make op null in list. Easier to walk list.
         *  Later we can throw as we add to index -> op map.
         *
         *  Note that I.2 R.2-2 will wipe out I.2 even though, technically, the
         *  inserted stuff would be before the replace range.  But, if you
         *  add tokens in front of a method body '{' and then delete the method
         *  body, I think the stuff before the '{' you added should disappear too.
         *
         *  Return a map from token index to operation.
         */
        protected virtual IDictionary<int, RewriteOperation> ReduceToSingleOperationPerIndex( IList<RewriteOperation> rewrites )
        {
            //System.out.println("rewrites="+rewrites);

            // WALK REPLACES
            for ( int i = 0; i < rewrites.Count; i++ )
            {
                RewriteOperation op = rewrites[i];
                if ( op == null )
                    continue;
                if ( !( op is ReplaceOp ) )
                    continue;
                ReplaceOp rop = (ReplaceOp)rewrites[i];
                // Wipe prior inserts within range
                var inserts = GetKindOfOps( rewrites, typeof( InsertBeforeOp ), i );
                for ( int j = 0; j < inserts.Count; j++ )
                {
                    InsertBeforeOp iop = (InsertBeforeOp)inserts[j];
                    if ( iop.index >= rop.index && iop.index <= rop.lastIndex )
                    {
                        // delete insert as it's a no-op.
                        rewrites[iop.instructionIndex] = null;
                    }
                }
                // Drop any prior replaces contained within
                var prevReplaces = GetKindOfOps( rewrites, typeof( ReplaceOp ), i );
                for ( int j = 0; j < prevReplaces.Count; j++ )
                {
                    ReplaceOp prevRop = (ReplaceOp)prevReplaces[j];
                    if ( prevRop.index >= rop.index && prevRop.lastIndex <= rop.lastIndex )
                    {
                        // delete replace as it's a no-op.
                        rewrites[prevRop.instructionIndex] = null;
                        continue;
                    }
                    // throw exception unless disjoint or identical
                    bool disjoint =
                        prevRop.lastIndex < rop.index || prevRop.index > rop.lastIndex;
                    bool same =
                        prevRop.index == rop.index && prevRop.lastIndex == rop.lastIndex;
                    if ( !disjoint && !same )
                    {
                        throw new ArgumentException( "replace op boundaries of " + rop +
                                                           " overlap with previous " + prevRop );
                    }
                }
            }

            // WALK INSERTS
            for ( int i = 0; i < rewrites.Count; i++ )
            {
                RewriteOperation op = (RewriteOperation)rewrites[i];
                if ( op == null )
                    continue;
                if ( !( op is InsertBeforeOp ) )
                    continue;
                InsertBeforeOp iop = (InsertBeforeOp)rewrites[i];
                // combine current insert with prior if any at same index
                var prevInserts = GetKindOfOps( rewrites, typeof( InsertBeforeOp ), i );
                for ( int j = 0; j < prevInserts.Count; j++ )
                {
                    InsertBeforeOp prevIop = (InsertBeforeOp)prevInserts[j];
                    if ( prevIop.index == iop.index )
                    { // combine objects
                        // convert to strings...we're in process of toString'ing
                        // whole token buffer so no lazy eval issue with any templates
                        iop.text = CatOpText( iop.text, prevIop.text );
                        // delete redundant prior insert
                        rewrites[prevIop.instructionIndex] = null;
                    }
                }
                // look for replaces where iop.index is in range; error
                var prevReplaces = GetKindOfOps( rewrites, typeof( ReplaceOp ), i );
                for ( int j = 0; j < prevReplaces.Count; j++ )
                {
                    ReplaceOp rop = (ReplaceOp)prevReplaces[j];
                    if ( iop.index == rop.index )
                    {
                        rop.text = CatOpText( iop.text, rop.text );
                        rewrites[i] = null;  // delete current insert
                        continue;
                    }
                    if ( iop.index >= rop.index && iop.index <= rop.lastIndex )
                    {
                        throw new ArgumentException( "insert op " + iop +
                                                           " within boundaries of previous " + rop );
                    }
                }
            }
            // System.out.println("rewrites after="+rewrites);
            IDictionary<int, RewriteOperation> m = new Dictionary<int, RewriteOperation>();
            for ( int i = 0; i < rewrites.Count; i++ )
            {
                RewriteOperation op = (RewriteOperation)rewrites[i];
                if ( op == null )
                    continue; // ignore deleted ops

                RewriteOperation existing;
                if ( m.TryGetValue( op.index, out existing ) && existing != null )
                {
                    throw new Exception( "should only be one op per index" );
                }
                m[op.index] = op;
            }
            //System.out.println("index to op: "+m);
            return m;
        }

        protected virtual string CatOpText( object a, object b )
        {
            return string.Concat( a, b );
        }
        protected virtual IList<RewriteOperation> GetKindOfOps( IList<RewriteOperation> rewrites, Type kind )
        {
            return GetKindOfOps( rewrites, kind, rewrites.Count );
        }

        /** <summary>Get all operations before an index of a particular kind</summary> */
        protected virtual IList<RewriteOperation> GetKindOfOps( IList<RewriteOperation> rewrites, Type kind, int before )
        {
            IList<RewriteOperation> ops = new List<RewriteOperation>();
            for ( int i = 0; i < before && i < rewrites.Count; i++ )
            {
                RewriteOperation op = rewrites[i];
                if ( op == null )
                    continue; // ignore deleted
                if ( op.GetType() == kind )
                    ops.Add( op );
            }
            return ops;
        }

        public virtual string ToDebugString()
        {
            return ToDebugString( MIN_TOKEN_INDEX, Size() - 1 );
        }

        public virtual string ToDebugString( int start, int end )
        {
            StringBuilder buf = new StringBuilder();
            for ( int i = start; i >= MIN_TOKEN_INDEX && i <= end && i < tokens.Count; i++ )
            {
                buf.Append( Get( i ) );
            }
            return buf.ToString();
        }
    }
}
