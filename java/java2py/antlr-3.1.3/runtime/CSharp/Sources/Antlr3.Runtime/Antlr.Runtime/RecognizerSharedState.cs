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
	using IDictionary = System.Collections.IDictionary;

	/// <summary>
	/// The set of fields needed by an abstract recognizer to recognize input
	/// and recover from errors 
	/// </summary>
	/// <remarks>
	/// As a separate state object, it can be shared among multiple grammars; 
	/// e.g., when one grammar imports another.
	/// These fields are publicly visible but the actual state pointer per 
	/// parser is protected.
	/// </remarks>
	public class RecognizerSharedState
	{
		/// <summary>
		/// Tracks the set of token types that can follow any rule invocation.
		/// Stack grows upwards.  When it hits the max, it grows 2x in size
		/// and keeps going.
		/// </summary>
		public BitSet[] following = new BitSet[BaseRecognizer.INITIAL_FOLLOW_STACK_SIZE];
		public int followingStackPointer = -1;

		/// <summary>
		/// This is true when we see an error and before having successfully
		/// matched a token.  Prevents generation of more than one error message
		/// per error.
		/// </summary>
		public bool errorRecovery = false;

		/// <summary>
		/// The index into the input stream where the last error occurred.
		/// </summary>
		/// <remarks>
		/// This is used to prevent infinite loops where an error is found
		/// but no token is consumed during recovery...another error is found,
		/// ad naseum.  This is a failsafe mechanism to guarantee that at least
		/// one token/tree node is consumed for two errors.
		/// </remarks>
		public int lastErrorIndex = -1;

		/// <summary>
		/// In lieu of a return value, this indicates that a rule or token
		/// has failed to match.  Reset to false upon valid token match.
		/// </summary>
		public bool failed = false;

		/// <summary>
		/// Did the recognizer encounter a syntax error?  Track how many.
		/// </summary>
		public int syntaxErrors = 0;

		/// <summary>
		/// If 0, no backtracking is going on.  Safe to exec actions etc...
		/// If >0 then it's the level of backtracking.
		/// </summary>
		public int backtracking = 0;

		/// <summary>
		/// An array[size num rules] of Map&lt;Integer,Integer&gt; that tracks
		/// the stop token index for each rule.
		/// </summary>
		/// <remarks>
		///  ruleMemo[ruleIndex] is the memoization table for ruleIndex.  
		///  For key ruleStartIndex, you get back the stop token for 
		///  associated rule or MEMO_RULE_FAILED.
		///  
		///  This is only used if rule memoization is on (which it is by default).
		///  </remarks>
		public IDictionary[] ruleMemo;


		#region Lexer Specific Members
		// LEXER FIELDS (must be in same state object to avoid casting
		//               constantly in generated code and Lexer object) :(


		/// <summary>
		/// Token object normally returned by NextToken() after matching lexer rules.
		/// </summary>
		/// <remarks>
		/// The goal of all lexer rules/methods is to create a token object.
		/// This is an instance variable as multiple rules may collaborate to
		/// create a single token.  nextToken will return this object after
		/// matching lexer rule(s).  If you subclass to allow multiple token
		/// emissions, then set this to the last token to be matched or
		/// something nonnull so that the auto token emit mechanism will not
		/// emit another token.
		/// </remarks>
		public IToken token;

		/// <summary>
		/// What character index in the stream did the current token start at?
		/// </summary>
		/// <remarks>
		/// Needed, for example, to get the text for current token.  Set at
		/// the start of nextToken.
		/// </remarks>
		public int tokenStartCharIndex = -1;

		/// <summary>
		/// The line on which the first character of the token resides
		/// </summary>
		public int tokenStartLine;

		/// <summary>The character position of first character within the line</summary>
		public int tokenStartCharPositionInLine;

		/// <summary>The channel number for the current token</summary>
		public int channel;

		/// <summary>The token type for the current token</summary>
		public int type;

		/// <summary>
		/// You can set the text for the current token to override what is in
		/// the input char buffer.  Use setText() or can set this instance var.
		/// </summary>
		public string text;

		#endregion
	}
}