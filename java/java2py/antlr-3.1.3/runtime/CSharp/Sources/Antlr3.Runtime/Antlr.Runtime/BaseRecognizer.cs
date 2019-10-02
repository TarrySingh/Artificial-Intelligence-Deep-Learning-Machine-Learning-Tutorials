/*
[The "BSD licence"]
Copyright (c) 2007-2008 Johannes Luber
Copyright (c) 2005-2007 Kunle Odutola
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code MUST RETAIN the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form MUST REPRODUCE the above copyright
   notice, this list of conditions and the following disclaimer in 
   the documentation and/or other materials provided with the 
   distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior WRITTEN permission.
4. Unless explicitly state otherwise, any contribution intentionally 
   submitted for inclusion in this work to the copyright owner or licensor
   shall be under the terms and conditions of this license, without any 
   additional terms or conditions.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


namespace Antlr.Runtime
{
	using System;
	using IList				= System.Collections.IList;
	using ArrayList			= System.Collections.ArrayList;
	using IDictionary		= System.Collections.IDictionary;
	using Hashtable			= System.Collections.Hashtable;
	using StackTrace		= System.Diagnostics.StackTrace;
	using StackFrame		= System.Diagnostics.StackFrame;
	using CollectionUtils = Antlr.Runtime.Collections.CollectionUtils;
	
	public delegate void SynPredPointer();

	/// <summary>
	/// A generic recognizer that can handle recognizers generated from
	/// lexer, parser, and tree grammars.  This is all the parsing
	/// support code essentially; most of it is error recovery stuff and
	/// backtracking.
	/// </summary>
	public abstract class BaseRecognizer
	{
		#region Constructors

		public BaseRecognizer()
		{
			state = new RecognizerSharedState();
		}

		public BaseRecognizer(RecognizerSharedState state)
		{
			if ( state==null ) {
				state = new RecognizerSharedState();
			}
			this.state = state;
		}

		#endregion


		#region Public API

		public virtual void BeginBacktrack(int level)
		{
		}

		public virtual void EndBacktrack(int level, bool successful)
		{
		}

		abstract public IIntStream Input { get; }

		public int BacktrackingLevel
		{
			get { return state.backtracking; }
			set { state.backtracking = value; }
		}

        /** Return whether or not a backtracking attempt failed. */
	    public bool Failed() { return state.failed; }

		/// <summary>Reset the parser's state. Subclasses must rewind the input stream.</summary>
		public virtual void Reset()
		{
			// wack everything related to error recovery
			if (state == null) {
				return; // no shared state work to do
			}
			state.followingStackPointer = -1;
			state.errorRecovery = false;
			state.lastErrorIndex = -1;
			state.failed = false;
			state.syntaxErrors = 0;
			// wack everything related to backtracking and memoization
			state.backtracking = 0;
			for (int i = 0; (state.ruleMemo != null) && (i < state.ruleMemo.Length); i++)
			{
				// wipe cache
				state.ruleMemo[i] = null;
			}
		}

		/// <summary>
		/// Match current input symbol against ttype.  Attempt
		/// single token insertion or deletion error recovery.  If
		/// that fails, throw MismatchedTokenException.
		/// </summary>
		/// <remarks>
		/// To turn off single token insertion or deletion error
		/// recovery, override RecoverFromMismatchedToken() and have it call
		/// pthrow an exception. See TreeParser.RecoverFromMismatchedToken().
		/// This way any error in a rule will cause an exception and
     	/// immediate exit from rule.  Rule would recover by resynchronizing
     	/// to the set of symbols that can follow rule ref.
		/// </remarks>
		public virtual object Match(IIntStream input, int ttype, BitSet follow) {
			object matchedSymbol = GetCurrentInputSymbol(input);
			if (input.LA(1) == ttype) {
				input.Consume();
				state.errorRecovery = false;
				state.failed = false;
				return matchedSymbol;
			}
			if (state.backtracking > 0) {
				state.failed = true;
				return matchedSymbol;
			}
			matchedSymbol = RecoverFromMismatchedToken(input, ttype, follow);
			return matchedSymbol;
		}

		/// <summary> Match the wildcard: in a symbol</summary>
		public virtual void MatchAny(IIntStream input)
		{
			state.errorRecovery = false;
			state.failed = false;
			input.Consume();
		}

		public bool MismatchIsUnwantedToken(IIntStream input, int ttype) {
			return input.LA(2) == ttype;
		}
	
		public bool MismatchIsMissingToken(IIntStream input, BitSet follow) {
			if (follow == null) {
				// we have no information about the follow; we can only consume
				// a single token and hope for the best
				return false;
			}

			// compute what can follow this grammar element reference
			if (follow.Member(Token.EOR_TOKEN_TYPE)) {
				BitSet viableTokensFollowingThisRule = ComputeContextSensitiveRuleFOLLOW();
				follow = follow.Or(viableTokensFollowingThisRule);
				if (state.followingStackPointer >= 0) { // remove EOR if we're not the start symbol
					follow.Remove(Token.EOR_TOKEN_TYPE);
				}
			}

			// if current token is consistent with what could come after set
			// then we know we're missing a token; error recovery is free to
			// "insert" the missing token

			// BitSet cannot handle negative numbers like -1 (EOF) so I leave EOR
			// in follow set to indicate that the fall of the start symbol is
			// in the set (EOF can follow).
			if ( follow.Member(input.LA(1)) || follow.Member(Token.EOR_TOKEN_TYPE) ) {
				return true;
			}
			return false;
		}

		/// <summary>
		/// Report a recognition problem.
		/// </summary>
		/// <remarks>
		/// This method sets errorRecovery to indicate the parser is recovering
		/// not parsing.  Once in recovery mode, no errors are generated.
		/// To get out of recovery mode, the parser must successfully Match
		/// a token (after a resync).  So it will go:
		/// 
		/// 1. error occurs
		/// 2. enter recovery mode, report error
		/// 3. consume until token found in resynch set
		/// 4. try to resume parsing
		/// 5. next Match() will reset errorRecovery mode
		///
		/// If you override, make sure to update syntaxErrors if you care about that.
		/// </remarks>
		public virtual void ReportError(RecognitionException e)
		{
			// if we've already reported an error and have not matched a token
			// yet successfully, don't report any errors.
			if (state.errorRecovery)
			{
				return;
			}
			state.syntaxErrors++; // don't count spurious
			state.errorRecovery = true;

			DisplayRecognitionError(this.TokenNames, e);
		}

		public virtual void DisplayRecognitionError(string[] tokenNames, RecognitionException e)
		{
			string hdr = GetErrorHeader(e);
			string msg = GetErrorMessage(e, tokenNames);
			EmitErrorMessage(hdr + " " + msg);
		}

		/// <summary>
		/// What error message should be generated for the various exception types?
		/// 
		/// Not very object-oriented code, but I like having all error message generation 
		/// within one method rather than spread among all of the exception classes. This 
		/// also makes it much easier for the exception handling because the exception 
		/// classes do not have to have pointers back to this object to access utility 
		/// routines and so on. Also, changing the message for an exception type would be 
		/// difficult because you would have to subclassing exception, but then somehow get 
		/// ANTLR to make those kinds of exception objects instead of the default.
		/// 
		/// This looks weird, but trust me--it makes the most sense in terms of flexibility.
		/// 
		/// For grammar debugging, you will want to override this to add more information 
		/// such as the stack frame with GetRuleInvocationStack(e, this.GetType().Fullname) 
		/// and, for no viable alts, the decision description and state etc...
		/// 
		/// Override this to change the message generated for one or more exception types.
		/// </summary>
		public virtual string GetErrorMessage(RecognitionException e, string[] tokenNames)
		{
			string msg = e.Message;
			if (e is UnwantedTokenException) {
				UnwantedTokenException ute = (UnwantedTokenException)e;
				string tokenName="<unknown>";
				if ( ute.Expecting== Token.EOF ) {
					tokenName = "EOF";
				}
				else {
					tokenName = tokenNames[ute.Expecting];
				}
				msg = "extraneous input " + GetTokenErrorDisplay(ute.UnexpectedToken) +
					" expecting " + tokenName;
			}
			else if (e is MissingTokenException) {
				MissingTokenException mte = (MissingTokenException)e;
				string tokenName="<unknown>";
				if (mte.Expecting == Token.EOF) {
					tokenName = "EOF";
				}
				else {
					tokenName = tokenNames[mte.Expecting];
				}
				msg = "missing " + tokenName + " at " + GetTokenErrorDisplay(e.Token);
			}
			else if (e is MismatchedTokenException)
			{
				MismatchedTokenException mte = (MismatchedTokenException)e;
				string tokenName = "<unknown>";
				if (mte.Expecting == Token.EOF)
				{
					tokenName = "EOF";
				}
				else
				{
					tokenName = tokenNames[mte.Expecting];
				}
				msg = "mismatched input " + GetTokenErrorDisplay(e.Token) + " expecting " + tokenName;

			}
			else if (e is MismatchedTreeNodeException)
			{
				MismatchedTreeNodeException mtne = (MismatchedTreeNodeException)e;
				string tokenName = "<unknown>";
				if (mtne.expecting == Token.EOF)
				{
					tokenName = "EOF";
				}
				else
				{
					tokenName = tokenNames[mtne.expecting];
				}
				// The ternary operator is only necessary because of a bug in the .NET framework
				msg = "mismatched tree node: " + ((mtne.Node!=null && mtne.Node.ToString()!=null) ?
					mtne.Node : string.Empty) + " expecting " + tokenName;
			}
			else if (e is NoViableAltException)
			{
				//NoViableAltException nvae = (NoViableAltException)e;
				// for development, can add "decision=<<"+nvae.grammarDecisionDescription+">>"
				// and "(decision="+nvae.decisionNumber+") and
				// "state "+nvae.stateNumber
				msg = "no viable alternative at input " + GetTokenErrorDisplay(e.Token);
			}
			else if (e is EarlyExitException)
			{
				//EarlyExitException eee = (EarlyExitException)e;
				// for development, can add "(decision="+eee.decisionNumber+")"
				msg = "required (...)+ loop did not match anything at input " + GetTokenErrorDisplay(e.Token);
			}
			else if (e is MismatchedSetException)
			{
				MismatchedSetException mse = (MismatchedSetException)e;
				msg = "mismatched input " + GetTokenErrorDisplay(e.Token) + " expecting set " + mse.expecting;
			}
			else if (e is MismatchedNotSetException)
			{
				MismatchedNotSetException mse = (MismatchedNotSetException)e;
				msg = "mismatched input " + GetTokenErrorDisplay(e.Token) + " expecting set " + mse.expecting;
			}
			else if (e is FailedPredicateException)
			{
				FailedPredicateException fpe = (FailedPredicateException)e;
				msg = "rule " + fpe.ruleName + " failed predicate: {" + fpe.predicateText + "}?";
			}
			return msg;
		}

		/// <summary>
		/// Get number of recognition errors (lexer, parser, tree parser).  Each
		/// recognizer tracks its own number.  So parser and lexer each have
		/// separate count.  Does not count the spurious errors found between
		/// an error and next valid token match
		///
		/// See also ReportError()
		/// </summary>
		public int NumberOfSyntaxErrors {
			get { return state.syntaxErrors; }
		}

		/// <summary>
		/// What is the error header, normally line/character position information?
		/// </summary>
		public virtual string GetErrorHeader(RecognitionException e)
		{
			return "line " + e.Line + ":" + e.CharPositionInLine;
		}

		/// <summary>
		/// How should a token be displayed in an error message? The default
		/// is to display just the text, but during development you might
		/// want to have a lot of information spit out.  Override in that case
		/// to use t.ToString() (which, for CommonToken, dumps everything about
		/// the token). This is better than forcing you to override a method in
		/// your token objects because you don't have to go modify your lexer
		/// so that it creates a new type.
		/// </summary>
		public virtual string GetTokenErrorDisplay(IToken t)
		{
			string s = t.Text;
			if (s == null)
			{
				if (t.Type == Token.EOF)
				{
					s = "<EOF>";
				}
				else
				{
					s = "<" + t.Type + ">";
				}
			}
			s = s.Replace("\n", "\\\\n");
			s = s.Replace("\r", "\\\\r");
			s = s.Replace("\t", "\\\\t");
			return "'" + s + "'";
		}

		/// <summary>
		/// Override this method to change where error messages go
		/// </summary>
		public virtual void EmitErrorMessage(string msg)
		{
			Console.Error.WriteLine(msg);
		}

		/// <summary>
		/// Recover from an error found on the input stream.  This is
		/// for NoViableAlt and mismatched symbol exceptions.  If you enable
		/// single token insertion and deletion, this will usually not
		/// handle mismatched symbol exceptions but there could be a mismatched
		/// token that the Match() routine could not recover from.
		/// </summary>
		public virtual void Recover(IIntStream input, RecognitionException re)
		{
			if (state.lastErrorIndex == input.Index())
			{
				// uh oh, another error at same token index; must be a case
				// where LT(1) is in the recovery token set so nothing is
				// consumed; consume a single token so at least to prevent
				// an infinite loop; this is a failsafe.
				input.Consume();
			}
			state.lastErrorIndex = input.Index();
			BitSet followSet = ComputeErrorRecoverySet();
			BeginResync();
			ConsumeUntil(input, followSet);
			EndResync();
		}

		/// <summary>A hook to listen in on the token consumption during error recovery.
		/// The DebugParser subclasses this to fire events to the listenter.
		/// </summary>
		public virtual void BeginResync()
		{
		}

		public virtual void EndResync()
		{
		}

		/// <summary>
		/// Attempt to Recover from a single missing or extra token.
		/// </summary>
		/// <remarks>
		/// EXTRA TOKEN
		/// 
		/// LA(1) is not what we are looking for.  If LA(2) has the right token,
		/// however, then assume LA(1) is some extra spurious token.  Delete it
		/// and LA(2) as if we were doing a normal Match(), which advances the
		/// input.
		/// 
		/// MISSING TOKEN
		/// 
		/// If current token is consistent with what could come after
		/// ttype then it is ok to "insert" the missing token, else throw
		/// exception For example, Input "i=(3;" is clearly missing the
		/// ')'.  When the parser returns from the nested call to expr, it
		/// will have call chain:
		/// 
		/// stat -> expr -> atom
		/// 
		/// and it will be trying to Match the ')' at this point in the
		/// derivation:
		/// 
		/// => ID '=' '(' INT ')' ('+' atom)* ';'
		/// ^
		/// Match() will see that ';' doesn't Match ')' and report a
		/// mismatched token error.  To Recover, it sees that LA(1)==';'
		/// is in the set of tokens that can follow the ')' token
		/// reference in rule atom.  It can assume that you forgot the ')'.
		/// </remarks>
		protected internal virtual Object RecoverFromMismatchedToken(IIntStream input, int ttype, BitSet follow) {
			RecognitionException e = null;
			// if next token is what we are looking for then "delete" this token
			if (MismatchIsUnwantedToken(input, ttype)) {
				e = new UnwantedTokenException(ttype, input);
				BeginResync();
				input.Consume(); // simply delete extra token
				EndResync();
				ReportError(e);  // report after consuming so AW sees the token in the exception
				// we want to return the token we're actually matching
				object matchedSymbol = GetCurrentInputSymbol(input);
				input.Consume(); // move past ttype token as if all were ok
				return matchedSymbol;
			}
			// can't recover with single token deletion, try insertion
			if (MismatchIsMissingToken(input, follow)) {
				object inserted = GetMissingSymbol(input, e, ttype, follow);
				e = new MissingTokenException(ttype, input, inserted);
				ReportError(e);  // report after inserting so AW sees the token in the exception
				return inserted;
			}
			// even that didn't work; must throw the exception
			e = new MismatchedTokenException(ttype, input);
			throw e;
		}

		/** Not currently used */
		public virtual object RecoverFromMismatchedSet(IIntStream input, RecognitionException e, BitSet follow) {
			if (MismatchIsMissingToken(input, follow)) {
				ReportError(e);
				// we don't know how to conjure up a token for sets yet
				return GetMissingSymbol(input, e, Token.INVALID_TOKEN_TYPE, follow);
			}
			// TODO do single token deletion like above for Token mismatch
			throw e;
		}

		public virtual void ConsumeUntil(IIntStream input, int tokenType)
		{
			int ttype = input.LA(1);
			while (ttype != Token.EOF && ttype != tokenType)
			{
				input.Consume();
				ttype = input.LA(1);
			}
		}

		/// <summary>Consume tokens until one matches the given token set </summary>
		public virtual void ConsumeUntil(IIntStream input, BitSet set)
		{
			int ttype = input.LA(1);
			while (ttype != Token.EOF && !set.Member(ttype))
			{
				input.Consume();
				ttype = input.LA(1);
			}
		}

		/// <summary>
		/// Returns List &lt;String&gt; of the rules in your parser instance
		/// leading up to a call to this method.  You could override if
		/// you want more details such as the file/line info of where
		/// in the parser source code a rule is invoked.
		/// </summary>
		/// <remarks>
		/// This is very useful for error messages and for context-sensitive
		/// error recovery.
		/// </remarks>
		public virtual IList GetRuleInvocationStack()
		{
			string parserClassName = GetType().FullName;
			return GetRuleInvocationStack(new System.Exception(), parserClassName);
		}

		/// <summary>
		/// A more general version of GetRuleInvocationStack where you can
		/// pass in, for example, a RecognitionException to get it's rule
		/// stack trace.  This routine is shared with all recognizers, hence,
		/// static.
		/// 
		/// TODO: move to a utility class or something; weird having lexer call this
		/// </summary>
		public static IList GetRuleInvocationStack(Exception e, string recognizerClassName)
		{
			IList rules = new ArrayList();
			StackTrace st = new StackTrace(e);
			int i = 0;
			for (i = st.FrameCount - 1; i >= 0; i--)
			{
				StackFrame sf = st.GetFrame(i);
				if (sf.GetMethod().DeclaringType.FullName.StartsWith("Antlr.Runtime."))
				{
					continue; // skip support code such as this method
				}
				if (sf.GetMethod().Name.Equals(NEXT_TOKEN_RULE_NAME))
				{
					continue;
				}
				if (!sf.GetMethod().DeclaringType.FullName.Equals(recognizerClassName))
				{
					continue; // must not be part of this parser
				}
				//kunle: rules.Add(sf.GetMethod().DeclaringType.FullName + "." + sf.GetMethod().Name);
				rules.Add(sf.GetMethod().Name);
			}
			return rules;
		}

		/// <summary>
		/// For debugging and other purposes, might want the grammar name.
		/// Have ANTLR generate an implementation for this property.
		/// </summary>
		/// <returns></returns>
		public virtual string GrammarFileName
		{
			get { return null; }
		}

		/// <summary>
		/// For debugging and other purposes, might want the source name.
		/// Have ANTLR provide a hook for this property.
		/// </summary>
		/// <returns>The source name</returns>
		public abstract string SourceName {
			get;
		}

		/// <summary>A convenience method for use most often with template rewrites.
		/// Convert a List&lt;Token&gt; to List&lt;String&gt;
		/// </summary>
		public virtual IList ToStrings(IList tokens)
		{
			if (tokens == null)
				return null;
			IList strings = new ArrayList(tokens.Count);
			for (int i = 0; i < tokens.Count; i++)
			{
				strings.Add(((IToken)tokens[i]).Text);
			}
			return strings;
		}

		/// <summary>
		/// Given a rule number and a start token index number, return
		/// MEMO_RULE_UNKNOWN if the rule has not parsed input starting from
		/// start index.  If this rule has parsed input starting from the
		/// start index before, then return where the rule stopped parsing.
		/// It returns the index of the last token matched by the rule.
		/// </summary>
		/// <remarks>
		/// For now we use a hashtable and just the slow Object-based one.
		/// Later, we can make a special one for ints and also one that
		/// tosses out data after we commit past input position i.
		/// </remarks>
		public virtual int GetRuleMemoization(int ruleIndex, int ruleStartIndex)
		{
			if (state.ruleMemo[ruleIndex] == null)
			{
				state.ruleMemo[ruleIndex] = new Hashtable();
			}
			object stopIndexI = state.ruleMemo[ruleIndex][(int)ruleStartIndex];
			if (stopIndexI == null)
			{
				return MEMO_RULE_UNKNOWN;
			}
			return (int)stopIndexI;
		}

		/// <summary>
		/// Has this rule already parsed input at the current index in the
		/// input stream?  Return the stop token index or MEMO_RULE_UNKNOWN.
		/// If we attempted but failed to parse properly before, return
		/// MEMO_RULE_FAILED.
		/// 
		/// This method has a side-effect: if we have seen this input for
		/// this rule and successfully parsed before, then seek ahead to
		/// 1 past the stop token matched for this rule last time.
		/// </summary>
		public virtual bool AlreadyParsedRule(IIntStream input, int ruleIndex)
		{
			int stopIndex = GetRuleMemoization(ruleIndex, input.Index());
			if (stopIndex == MEMO_RULE_UNKNOWN)
			{
				return false;
			}
			if (stopIndex == MEMO_RULE_FAILED)
			{
				state.failed = true;
			}
			else
			{
				input.Seek(stopIndex + 1); // jump to one past stop token
			}
			return true;
		}

		/// <summary>
		/// Record whether or not this rule parsed the input at this position
		/// successfully.  Use a standard hashtable for now.
		/// </summary>
		public virtual void Memoize(IIntStream input, int ruleIndex, int ruleStartIndex)
		{
			int stopTokenIndex = state.failed ? MEMO_RULE_FAILED : input.Index() - 1;
			if (state.ruleMemo[ruleIndex] != null)
			{
				state.ruleMemo[ruleIndex][(int)ruleStartIndex] = (int)stopTokenIndex;
			}
		}

		/// <summary>
		/// Return how many rule/input-index pairs there are in total.
		///  TODO: this includes synpreds. :(
		/// </summary>
		/// <returns></returns>
		public int GetRuleMemoizationCacheSize()
		{
			int n = 0;
			for (int i = 0; state.ruleMemo != null && i < state.ruleMemo.Length; i++)
			{
				IDictionary ruleMap = state.ruleMemo[i];
				if (ruleMap != null)
				{
					n += ruleMap.Count; // how many input indexes are recorded?
				}
			}
			return n;
		}

		public virtual void TraceIn(string ruleName, int ruleIndex, object inputSymbol)
		{
			Console.Out.Write("enter " + ruleName + " " + inputSymbol);
			if (state.backtracking > 0)
			{
				Console.Out.Write(" backtracking=" + state.backtracking);
			}
			Console.Out.WriteLine();
		}

		public virtual void TraceOut(string ruleName, int ruleIndex, object inputSymbol) {
			Console.Out.Write("exit " + ruleName + " " + inputSymbol);
			if (state.backtracking > 0) {
				Console.Out.Write(" backtracking=" + state.backtracking);
				if (state.failed) {
					Console.Out.WriteLine(" failed" + state.failed);
				} else {
					Console.Out.WriteLine(" succeeded" + state.failed);
				}
			}
			Console.Out.WriteLine();
		}


		/// <summary>
		/// Used to print out token names like ID during debugging and
		/// error reporting.  The generated parsers implement a method
		/// that overrides this to point to their string[] tokenNames.
		/// </summary>
		virtual public string[] TokenNames
		{
			get { return null; }
		}

		#endregion


		#region Helper Methods

		/// <summary>
		/// Factor out what to do upon token mismatch so tree parsers can behave
		/// differently.  Override and call RecoverFromMismatchedToken()
	 	/// to get single token insertion and deletion. Use this to turn off
		/// single token insertion and deletion. Override mismatchRecover
		/// to call this instead.
	    /// 
	    /// TODO: fix this comment, mismatchRecover doesn't exist, for example
		/// </summary>
		/*
		protected internal virtual void Mismatch(IIntStream input, int ttype, BitSet follow)
		{
			if ( MismatchIsUnwantedToken(input, ttype) ) {
				throw new UnwantedTokenException(ttype, input);
			}
			else if ( MismatchIsMissingToken(input, follow) ) {
				throw new MissingTokenException(ttype, input, null);
			}
			throw new MismatchedTokenException(ttype, input);
		}*/
	
		/*  Compute the error recovery set for the current rule.  During
		*  rule invocation, the parser pushes the set of tokens that can
		*  follow that rule reference on the stack; this amounts to
		*  computing FIRST of what follows the rule reference in the
		*  enclosing rule. This local follow set only includes tokens
		*  from within the rule; i.e., the FIRST computation done by
		*  ANTLR stops at the end of a rule.
		*
		*  EXAMPLE
		*
		*  When you find a "no viable alt exception", the input is not
		*  consistent with any of the alternatives for rule r.  The best
		*  thing to do is to consume tokens until you see something that
		*  can legally follow a call to r *or* any rule that called r.
		*  You don't want the exact set of viable next tokens because the
		*  input might just be missing a token--you might consume the
		*  rest of the input looking for one of the missing tokens.
		*
		*  Consider grammar:
		*
		*  a : '[' b ']'
		*    | '(' b ')'
		*    ;
		*  b : c '^' INT ;
		*  c : ID
		*    | INT
		*    ;
		*
		*  At each rule invocation, the set of tokens that could follow
		*  that rule is pushed on a stack.  Here are the various "local"
		*  follow sets:
		*
		*  FOLLOW(b1_in_a) = FIRST(']') = ']'
		*  FOLLOW(b2_in_a) = FIRST(')') = ')'
		*  FOLLOW(c_in_b) = FIRST('^') = '^'
		*
		*  Upon erroneous input "[]", the call chain is
		*
		*  a -> b -> c
		*
		*  and, hence, the follow context stack is:
		*
		*  depth  local follow set     after call to rule
		*    0         <EOF>                    a (from main())
		*    1          ']'                     b
		*    3          '^'                     c
		*
		*  Notice that ')' is not included, because b would have to have
		*  been called from a different context in rule a for ')' to be
		*  included.
		*
		*  For error recovery, we cannot consider FOLLOW(c)
		*  (context-sensitive or otherwise).  We need the combined set of
		*  all context-sensitive FOLLOW sets--the set of all tokens that
		*  could follow any reference in the call chain.  We need to
		*  resync to one of those tokens.  Note that FOLLOW(c)='^' and if
		*  we resync'd to that token, we'd consume until EOF.  We need to
		*  sync to context-sensitive FOLLOWs for a, b, and c: {']','^'}.
		*  In this case, for input "[]", LA(1) is in this set so we would
		*  not consume anything and after printing an error rule c would
		*  return normally.  It would not find the required '^' though.
		*  At this point, it gets a mismatched token error and throws an
		*  exception (since LA(1) is not in the viable following token
		*  set).  The rule exception handler tries to Recover, but finds
		*  the same recovery set and doesn't consume anything.  Rule b
		*  exits normally returning to rule a.  Now it finds the ']' (and
		*  with the successful Match exits errorRecovery mode).
		*
		*  So, you cna see that the parser walks up call chain looking
		*  for the token that was a member of the recovery set.
		*
		*  Errors are not generated in errorRecovery mode.
		*
		*  ANTLR's error recovery mechanism is based upon original ideas:
		*
		*  "Algorithms + Data Structures = Programs" by Niklaus Wirth
		*
		*  and
		*
		*  "A note on error recovery in recursive descent parsers":
		*  http://portal.acm.org/citation.cfm?id=947902.947905
		*
		*  Later, Josef Grosch had some good ideas:
		*
		*  "Efficient and Comfortable Error Recovery in Recursive Descent
		*  Parsers":
		*  ftp://www.cocolab.com/products/cocktail/doca4.ps/ell.ps.zip
		*
		*  Like Grosch I implemented local FOLLOW sets that are combined
		*  at run-time upon error to avoid overhead during parsing.
		*/
		protected internal virtual BitSet ComputeErrorRecoverySet()
		{
			return CombineFollows(false);
		}
		
		/// <summary>Compute the context-sensitive FOLLOW set for current rule.
		/// This is set of token types that can follow a specific rule
		/// reference given a specific call chain.  You get the set of
		/// viable tokens that can possibly come next (lookahead depth 1)
		/// given the current call chain.  Contrast this with the
		/// definition of plain FOLLOW for rule r:
		/// 
		/// FOLLOW(r)={x | S=>*alpha r beta in G and x in FIRST(beta)}
		/// 
		/// where x in T* and alpha, beta in V*; T is set of terminals and
		/// V is the set of terminals and nonterminals.  In other words,
		/// FOLLOW(r) is the set of all tokens that can possibly follow
		/// references to r in *any* sentential form (context).  At
		/// runtime, however, we know precisely which context applies as
		/// we have the call chain.  We may compute the exact (rather
		/// than covering superset) set of following tokens.
		/// 
		/// For example, consider grammar:
		/// 
		/// stat : ID '=' expr ';'      // FOLLOW(stat)=={EOF}
		/// | "return" expr '.'
		/// ;
		/// expr : atom ('+' atom)* ;   // FOLLOW(expr)=={';','.',')'}
		/// atom : INT                  // FOLLOW(atom)=={'+',')',';','.'}
		/// | '(' expr ')'
		/// ;
		/// 
		/// The FOLLOW sets are all inclusive whereas context-sensitive
		/// FOLLOW sets are precisely what could follow a rule reference.
		/// For input input "i=(3);", here is the derivation:
		/// 
		/// stat => ID '=' expr ';'
		/// => ID '=' atom ('+' atom)* ';'
		/// => ID '=' '(' expr ')' ('+' atom)* ';'
		/// => ID '=' '(' atom ')' ('+' atom)* ';'
		/// => ID '=' '(' INT ')' ('+' atom)* ';'
		/// => ID '=' '(' INT ')' ';'
		/// 
		/// At the "3" token, you'd have a call chain of
		/// 
		/// stat -> expr -> atom -> expr -> atom
		/// 
		/// What can follow that specific nested ref to atom?  Exactly ')'
		/// as you can see by looking at the derivation of this specific
		/// input.  Contrast this with the FOLLOW(atom)={'+',')',';','.'}.
		/// 
		/// You want the exact viable token set when recovering from a
		/// token mismatch.  Upon token mismatch, if LA(1) is member of
		/// the viable next token set, then you know there is most likely
		/// a missing token in the input stream.  "Insert" one by just not
		/// throwing an exception.
		/// </summary>
		protected internal virtual BitSet ComputeContextSensitiveRuleFOLLOW()
		{
			return CombineFollows(true);
		}
		
		protected internal virtual BitSet CombineFollows(bool exact)
		{
			int top = state.followingStackPointer;
			BitSet followSet = new BitSet();
			for (int i = top; i >= 0; i--)
			{
				BitSet localFollowSet = (BitSet)state.following[i];
				followSet.OrInPlace(localFollowSet);
				if (exact) {
					// can we see end of rule?
					if (localFollowSet.Member(Token.EOR_TOKEN_TYPE)) {
						// Only leave EOR in set if at top (start rule); this lets
						// us know if have to include follow(start rule); i.e., EOF
						if (i > 0) {
							followSet.Remove(Token.EOR_TOKEN_TYPE);
						}
					}
					else { // can't see end of rule, quit
						break;
					}
				}
			}
			return followSet;
		}
		
		/// <summary>
		/// Match needs to return the current input symbol, which gets put
		/// into the label for the associated token ref; e.g., x=ID.  Token
		/// and tree parsers need to return different objects. Rather than test
		/// for input stream type or change the IntStream interface, I use
		/// a simple method to ask the recognizer to tell me what the current
		/// input symbol is.
		/// </summary>
		/// <remarks>This is ignored for lexers.</remarks>
		protected virtual object GetCurrentInputSymbol(IIntStream input) {
			return null;
		}

		/// <summary>
		/// Conjure up a missing token during error recovery.
		/// </summary>
		/// <remarks>
		/// The recognizer attempts to recover from single missing
		/// symbols. But, actions might refer to that missing symbol.
		/// For example, x=ID {f($x);}. The action clearly assumes
		/// that there has been an identifier matched previously and that
		/// $x points at that token. If that token is missing, but
		/// the next token in the stream is what we want we assume that
		/// this token is missing and we keep going. Because we
		/// have to return some token to replace the missing token,
		/// we have to conjure one up. This method gives the user control
		/// over the tokens returned for missing tokens. Mostly,
		/// you will want to create something special for identifier
		/// tokens. For literals such as '{' and ',', the default
		/// action in the parser or tree parser works. It simply creates
		/// a CommonToken of the appropriate type. The text will be the token.
		/// If you change what tokens must be created by the lexer,
		/// override this method to create the appropriate tokens.
		/// </remarks>
		protected virtual object GetMissingSymbol(IIntStream input,
										  RecognitionException e,
										  int expectedTokenType,
										  BitSet follow) {
			return null;
		}
		
		/// <summary>
		/// Push a rule's follow set using our own hardcoded stack 
		/// </summary>
		/// <param name="fset"></param>
		protected void PushFollow(BitSet fset)
		{
			if ((state.followingStackPointer + 1) >= state.following.Length)
			{
				BitSet[] f = new BitSet[state.following.Length * 2];
				Array.Copy(state.following, 0, f, 0, state.following.Length);
				state.following = f;
			}
			state.following[++state.followingStackPointer] = fset;
		}

		#endregion


		#region Data Members

		public const int MEMO_RULE_FAILED = -2;
		public const int MEMO_RULE_UNKNOWN = -1;
		public const int INITIAL_FOLLOW_STACK_SIZE = 100;

		// copies from Token object for convenience in actions
		public const int DEFAULT_TOKEN_CHANNEL = Token.DEFAULT_CHANNEL;
		public const int HIDDEN = Token.HIDDEN_CHANNEL;

		public static readonly string NEXT_TOKEN_RULE_NAME = "nextToken";

		/// <summary>
		/// An externalized representation of the - shareable - internal state of 
		/// this lexer, parser or tree parser.
		/// </summary>
		/// <remarks>
		/// The state of a lexer, parser, or tree parser are collected into 
		/// external state objects so that the state can be shared. This sharing 
		/// is needed to have one grammar import others and share same error 
		/// variables and other state variables.  It's a kind of explicit multiple
		/// inheritance via delegation of methods and shared state.
		/// </remarks>
		protected internal RecognizerSharedState state;

		#endregion
	}
}