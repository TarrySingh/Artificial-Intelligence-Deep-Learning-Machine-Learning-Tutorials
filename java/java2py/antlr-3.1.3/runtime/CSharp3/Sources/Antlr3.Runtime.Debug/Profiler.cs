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
    using System.Collections.Generic;
    using System.Linq;
    using Antlr.Runtime.JavaExtensions;

    using Array = System.Array;
    using Console = System.Console;
    using IOException = System.IO.IOException;
    using Stats = Antlr.Runtime.Misc.Stats;
    using StringBuilder = System.Text.StringBuilder;

    /** <summary>Using the debug event interface, track what is happening in the parser
     *  and record statistics about the runtime.
     */
    public class Profiler : BlankDebugEventListener
    {
        /** <summary>Because I may change the stats, I need to track that for later
         *  computations to be consistent.
         */
        public const string Version = "2";
        public const string RUNTIME_STATS_FILENAME = "runtime.stats";
        public const int NUM_RUNTIME_STATS = 29;

        public DebugParser parser = null;

        #region working variables

        protected int ruleLevel = 0;
        protected int decisionLevel = 0;
        protected int maxLookaheadInCurrentDecision = 0;
        protected CommonToken lastTokenConsumed = null;

        protected Stack<int> lookaheadStack = new Stack<int>();

        #endregion


        #region stats variables

        public int numRuleInvocations = 0;
        public int numGuessingRuleInvocations = 0;
        public int maxRuleInvocationDepth = 0;
        public int numFixedDecisions = 0;
        public int numCyclicDecisions = 0;
        public int numBacktrackDecisions = 0;
        public int[] decisionMaxFixedLookaheads = new int[200]; // TODO: make List
        public int[] decisionMaxCyclicLookaheads = new int[200];
        public List<int> decisionMaxSynPredLookaheads = new List<int>();
        public int numHiddenTokens = 0;
        public int numCharsMatched = 0;
        public int numHiddenCharsMatched = 0;
        public int numSemanticPredicates = 0;
        public int numSyntacticPredicates = 0;
        protected int numberReportedErrors = 0;
        public int numMemoizationCacheMisses = 0;
        public int numMemoizationCacheHits = 0;
        public int numMemoizationCacheEntries = 0;

        #endregion

        public Profiler()
        {
        }

        public Profiler( DebugParser parser )
        {
            this.parser = parser;
        }

        public override void EnterRule( string grammarFileName, string ruleName )
        {
            //System.out.println("enterRule "+ruleName);
            ruleLevel++;
            numRuleInvocations++;
            if ( ruleLevel > maxRuleInvocationDepth )
            {
                maxRuleInvocationDepth = ruleLevel;
            }

        }

        /** <summary>
         *  Track memoization; this is not part of standard debug interface
         *  but is triggered by profiling.  Code gen inserts an override
         *  for this method in the recognizer, which triggers this method.
         *  </summary>
         */
        public virtual void ExamineRuleMemoization( IIntStream input,
                                           int ruleIndex,
                                           string ruleName )
        {
            //System.out.println("examine memo "+ruleName);
            int stopIndex = parser.GetRuleMemoization( ruleIndex, input.Index );
            if ( stopIndex == BaseRecognizer.MEMO_RULE_UNKNOWN )
            {
                //System.out.println("rule "+ruleIndex+" missed @ "+input.index());
                numMemoizationCacheMisses++;
                numGuessingRuleInvocations++; // we'll have to enter
            }
            else
            {
                // regardless of rule success/failure, if in cache, we have a cache hit
                //System.out.println("rule "+ruleIndex+" hit @ "+input.index());
                numMemoizationCacheHits++;
            }
        }

        public virtual void Memoize( IIntStream input,
                            int ruleIndex,
                            int ruleStartIndex,
                            string ruleName )
        {
            // count how many entries go into table
            //System.out.println("memoize "+ruleName);
            numMemoizationCacheEntries++;
        }

        public override void ExitRule( string grammarFileName, string ruleName )
        {
            ruleLevel--;
        }

        public override void EnterDecision( int decisionNumber )
        {
            decisionLevel++;
            int startingLookaheadIndex = parser.TokenStream.Index;
            //System.out.println("enterDecision "+decisionNumber+" @ index "+startingLookaheadIndex);
            lookaheadStack.Push( startingLookaheadIndex );
        }

        public override void ExitDecision( int decisionNumber )
        {
            //System.out.println("exitDecision "+decisionNumber);
            // track how many of acyclic, cyclic here as we don't know what kind
            // yet in enterDecision event.
            if ( parser.isCyclicDecision )
            {
                numCyclicDecisions++;
            }
            else
            {
                numFixedDecisions++;
            }
            lookaheadStack.Pop(); // pop lookahead depth counter
            decisionLevel--;
            if ( parser.isCyclicDecision )
            {
                if ( numCyclicDecisions >= decisionMaxCyclicLookaheads.Length )
                {
                    int[] bigger = new int[decisionMaxCyclicLookaheads.Length * 2];
                    Array.Copy( decisionMaxCyclicLookaheads, bigger, decisionMaxCyclicLookaheads.Length );
                    decisionMaxCyclicLookaheads = bigger;
                }
                decisionMaxCyclicLookaheads[numCyclicDecisions - 1] = maxLookaheadInCurrentDecision;
            }
            else
            {
                if ( numFixedDecisions >= decisionMaxFixedLookaheads.Length )
                {
                    int[] bigger = new int[decisionMaxFixedLookaheads.Length * 2];
                    Array.Copy( decisionMaxFixedLookaheads, bigger, decisionMaxFixedLookaheads.Length );
                    decisionMaxFixedLookaheads = bigger;
                }
                decisionMaxFixedLookaheads[numFixedDecisions - 1] = maxLookaheadInCurrentDecision;
            }
            parser.isCyclicDecision = false; // can't nest so just reset to false
            maxLookaheadInCurrentDecision = 0;
        }

        public override void ConsumeToken( IToken token )
        {
            //System.out.println("consume token "+token);
            lastTokenConsumed = (CommonToken)token;
        }

        /** <summary>
         *  The parser is in a decision if the decision depth > 0.  This
         *  works for backtracking also, which can have nested decisions.
         *  </summary>
         */
        public virtual bool InDecision
        {
            get
            {
                return decisionLevel > 0;
            }
        }

        public override void ConsumeHiddenToken( IToken token )
        {
            //System.out.println("consume hidden token "+token);
            lastTokenConsumed = (CommonToken)token;
        }

        /** <summary>Track refs to lookahead if in a fixed/nonfixed decision.</summary> */
        public override void LT( int i, IToken t )
        {
            if ( InDecision )
            {
                // get starting index off stack
                int startingIndex = lookaheadStack.Peek();
                // compute lookahead depth
                int thisRefIndex = parser.TokenStream.Index;
                int numHidden = GetNumberOfHiddenTokens( startingIndex, thisRefIndex );
                int depth = i + thisRefIndex - startingIndex - numHidden;
                /*
                System.out.println("LT("+i+") @ index "+thisRefIndex+" is depth "+depth+
                    " max is "+maxLookaheadInCurrentDecision);
                */
                if ( depth > maxLookaheadInCurrentDecision )
                {
                    maxLookaheadInCurrentDecision = depth;
                }
            }
        }

        /** <summary>
         *  Track backtracking decisions.  You'll see a fixed or cyclic decision
         *  and then a backtrack.
         *  </summary>
         *
         *  <remarks>
         *      enter rule
         *      ...
         *      enter decision
         *      LA and possibly consumes (for cyclic DFAs)
         *      begin backtrack level
         *      mark m
         *      rewind m
         *      end backtrack level, success
         *      exit decision
         *      ...
         *      exit rule
         *  </remarks>
         */
        public override void BeginBacktrack( int level )
        {
            //System.out.println("enter backtrack "+level);
            numBacktrackDecisions++;
        }

        /** <summary>Successful or not, track how much lookahead synpreds use</summary> */
        public override void EndBacktrack( int level, bool successful )
        {
            //System.out.println("exit backtrack "+level+": "+successful);
            decisionMaxSynPredLookaheads.Add(
                maxLookaheadInCurrentDecision
            );
        }

#if false
        public void mark( int marker )
        {
            int i = parser.TokenStream.Index;
            JSystem.@out.println( "mark @ index " + i );
            synPredLookaheadStack.Push( i );
        }

        public void rewind( int marker )
        {
            // pop starting index off stack
            int startingIndex = synPredLookaheadStack.Pop();
            // compute lookahead depth
            int stopIndex = parser.TokenStream.Index;
            JSystem.@out.println( "rewind @ index " + stopIndex );
            int depth = stopIndex - startingIndex;
            JSystem.@out.println( "depth of lookahead for synpred: " + depth );
            decisionMaxSynPredLookaheads.Add( depth );
        }
#endif

        public override void RecognitionException( RecognitionException e )
        {
            numberReportedErrors++;
        }

        public override void SemanticPredicate( bool result, string predicate )
        {
            if ( InDecision )
            {
                numSemanticPredicates++;
            }
        }

        public override void Terminate()
        {
            string stats = ToNotifyString();
            try
            {
                Stats.WriteReport( RUNTIME_STATS_FILENAME, stats );
            }
            catch ( IOException ioe )
            {
                Console.Error.WriteLine( ioe );
                ioe.PrintStackTrace( Console.Error );
            }
            Console.Out.WriteLine( ToString( stats ) );
        }

        public virtual void SetParser( DebugParser parser )
        {
            this.parser = parser;
        }

        #region Reporting

        public virtual string ToNotifyString()
        {
            ITokenStream input = parser.TokenStream;
            for ( int i = 0; i < input.Size() && lastTokenConsumed != null && i <= lastTokenConsumed.TokenIndex; i++ )
            {
                IToken t = input.Get( i );
                if ( t.Channel != TokenConstants.DEFAULT_CHANNEL )
                {
                    numHiddenTokens++;
                    numHiddenCharsMatched += t.Text.Length;
                }
            }
            numCharsMatched = lastTokenConsumed.StopIndex + 1;
            decisionMaxFixedLookaheads = Trim( decisionMaxFixedLookaheads, numFixedDecisions );
            decisionMaxCyclicLookaheads = Trim( decisionMaxCyclicLookaheads, numCyclicDecisions );
            StringBuilder buf = new StringBuilder();
            buf.Append( Version );
            buf.Append( '\t' );
            buf.Append( parser.GetType().Name );
            buf.Append( '\t' );
            buf.Append( numRuleInvocations );
            buf.Append( '\t' );
            buf.Append( maxRuleInvocationDepth );
            buf.Append( '\t' );
            buf.Append( numFixedDecisions );
            buf.Append( '\t' );
            buf.Append( decisionMaxFixedLookaheads.DefaultIfEmpty( int.MaxValue ).Min() );
            buf.Append( '\t' );
            buf.Append( decisionMaxFixedLookaheads.DefaultIfEmpty( int.MinValue ).Max() );
            buf.Append( '\t' );
            buf.Append( decisionMaxFixedLookaheads.DefaultIfEmpty( 0 ).Average() );
            buf.Append( '\t' );
            buf.Append( Stats.Stddev( decisionMaxFixedLookaheads ) );
            buf.Append( '\t' );
            buf.Append( numCyclicDecisions );
            buf.Append( '\t' );
            buf.Append( decisionMaxCyclicLookaheads.DefaultIfEmpty( int.MaxValue ).Min() );
            buf.Append( '\t' );
            buf.Append( decisionMaxCyclicLookaheads.DefaultIfEmpty( int.MinValue ).Max() );
            buf.Append( '\t' );
            buf.Append( decisionMaxCyclicLookaheads.DefaultIfEmpty( 0 ).Average() );
            buf.Append( '\t' );
            buf.Append( Stats.Stddev( decisionMaxCyclicLookaheads ) );
            buf.Append( '\t' );
            buf.Append( numBacktrackDecisions );
            buf.Append( '\t' );
            buf.Append( decisionMaxSynPredLookaheads.DefaultIfEmpty( int.MaxValue ).Min() );
            buf.Append( '\t' );
            buf.Append( decisionMaxSynPredLookaheads.DefaultIfEmpty( int.MinValue ).Max() );
            buf.Append( '\t' );
            buf.Append( decisionMaxSynPredLookaheads.DefaultIfEmpty( 0 ).Average() );
            buf.Append( '\t' );
            buf.Append( Stats.Stddev( decisionMaxSynPredLookaheads ) );
            buf.Append( '\t' );
            buf.Append( numSemanticPredicates );
            buf.Append( '\t' );
            buf.Append( parser.TokenStream.Size() );
            buf.Append( '\t' );
            buf.Append( numHiddenTokens );
            buf.Append( '\t' );
            buf.Append( numCharsMatched );
            buf.Append( '\t' );
            buf.Append( numHiddenCharsMatched );
            buf.Append( '\t' );
            buf.Append( numberReportedErrors );
            buf.Append( '\t' );
            buf.Append( numMemoizationCacheHits );
            buf.Append( '\t' );
            buf.Append( numMemoizationCacheMisses );
            buf.Append( '\t' );
            buf.Append( numGuessingRuleInvocations );
            buf.Append( '\t' );
            buf.Append( numMemoizationCacheEntries );
            return buf.ToString();
        }

        public override string ToString()
        {
            return ToString( ToNotifyString() );
        }

        protected static string[] DecodeReportData( string data )
        {
            string[] fields = new string[NUM_RUNTIME_STATS];
            StringTokenizer st = new StringTokenizer( data, "\t" );
            int i = 0;
            while ( st.hasMoreTokens() )
            {
                fields[i] = st.nextToken();
                i++;
            }
            if ( i != NUM_RUNTIME_STATS )
            {
                return null;
            }
            return fields;
        }

        public static string ToString( string notifyDataLine )
        {
            string[] fields = DecodeReportData( notifyDataLine );
            if ( fields == null )
            {
                return null;
            }
            StringBuilder buf = new StringBuilder();
            buf.Append( "ANTLR Runtime Report; Profile Version " );
            buf.Append( fields[0] );
            buf.Append( '\n' );
            buf.Append( "parser name " );
            buf.Append( fields[1] );
            buf.Append( '\n' );
            buf.Append( "Number of rule invocations " );
            buf.Append( fields[2] );
            buf.Append( '\n' );
            buf.Append( "Number of rule invocations in \"guessing\" mode " );
            buf.Append( fields[27] );
            buf.Append( '\n' );
            buf.Append( "max rule invocation nesting depth " );
            buf.Append( fields[3] );
            buf.Append( '\n' );
            buf.Append( "number of fixed lookahead decisions " );
            buf.Append( fields[4] );
            buf.Append( '\n' );
            buf.Append( "min lookahead used in a fixed lookahead decision " );
            buf.Append( fields[5] );
            buf.Append( '\n' );
            buf.Append( "max lookahead used in a fixed lookahead decision " );
            buf.Append( fields[6] );
            buf.Append( '\n' );
            buf.Append( "average lookahead depth used in fixed lookahead decisions " );
            buf.Append( fields[7] );
            buf.Append( '\n' );
            buf.Append( "standard deviation of depth used in fixed lookahead decisions " );
            buf.Append( fields[8] );
            buf.Append( '\n' );
            buf.Append( "number of arbitrary lookahead decisions " );
            buf.Append( fields[9] );
            buf.Append( '\n' );
            buf.Append( "min lookahead used in an arbitrary lookahead decision " );
            buf.Append( fields[10] );
            buf.Append( '\n' );
            buf.Append( "max lookahead used in an arbitrary lookahead decision " );
            buf.Append( fields[11] );
            buf.Append( '\n' );
            buf.Append( "average lookahead depth used in arbitrary lookahead decisions " );
            buf.Append( fields[12] );
            buf.Append( '\n' );
            buf.Append( "standard deviation of depth used in arbitrary lookahead decisions " );
            buf.Append( fields[13] );
            buf.Append( '\n' );
            buf.Append( "number of evaluated syntactic predicates " );
            buf.Append( fields[14] );
            buf.Append( '\n' );
            buf.Append( "min lookahead used in a syntactic predicate " );
            buf.Append( fields[15] );
            buf.Append( '\n' );
            buf.Append( "max lookahead used in a syntactic predicate " );
            buf.Append( fields[16] );
            buf.Append( '\n' );
            buf.Append( "average lookahead depth used in syntactic predicates " );
            buf.Append( fields[17] );
            buf.Append( '\n' );
            buf.Append( "standard deviation of depth used in syntactic predicates " );
            buf.Append( fields[18] );
            buf.Append( '\n' );
            buf.Append( "rule memoization cache size " );
            buf.Append( fields[28] );
            buf.Append( '\n' );
            buf.Append( "number of rule memoization cache hits " );
            buf.Append( fields[25] );
            buf.Append( '\n' );
            buf.Append( "number of rule memoization cache misses " );
            buf.Append( fields[26] );
            buf.Append( '\n' );
            buf.Append( "number of evaluated semantic predicates " );
            buf.Append( fields[19] );
            buf.Append( '\n' );
            buf.Append( "number of tokens " );
            buf.Append( fields[20] );
            buf.Append( '\n' );
            buf.Append( "number of hidden tokens " );
            buf.Append( fields[21] );
            buf.Append( '\n' );
            buf.Append( "number of char " );
            buf.Append( fields[22] );
            buf.Append( '\n' );
            buf.Append( "number of hidden char " );
            buf.Append( fields[23] );
            buf.Append( '\n' );
            buf.Append( "number of syntax errors " );
            buf.Append( fields[24] );
            buf.Append( '\n' );
            return buf.ToString();
        }

        #endregion

        protected virtual int[] Trim( int[] X, int n )
        {
            if ( n < X.Length )
            {
                int[] trimmed = new int[n];
                Array.Copy( X, trimmed, n );
                X = trimmed;
            }
            return X;
        }

        /** <summary>Get num hidden tokens between i..j inclusive</summary> */
        public virtual int GetNumberOfHiddenTokens( int i, int j )
        {
            int n = 0;
            ITokenStream input = parser.TokenStream;
            for ( int ti = i; ti < input.Size() && ti <= j; ti++ )
            {
                IToken t = input.Get( ti );
                if ( t.Channel != TokenConstants.DEFAULT_CHANNEL )
                {
                    n++;
                }
            }
            return n;
        }
    }
}
