/*
[The "BSD licence"]
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
	
	/// <summary>A stream of tokens accessing tokens from a TokenSource </summary>
	public interface ITokenStream : IIntStream
	{
		/// <summary>
		/// Get Token at current input pointer + i ahead (where i=1 is next 
		/// Token).
		/// i &lt; 0 indicates tokens in the past.  So -1 is previous token and -2 is
		/// two tokens ago. LT(0) is undefined.  For i>=n, return Token.EOFToken.
		/// Return null for LT(0) and any index that results in an absolute address
		/// that is negative.
		/// </summary>
		IToken LT(int k);
		
		/// <summary>
		/// Get a token at an absolute index i; 0..n-1.  This is really only
		/// needed for profiling and debugging and token stream rewriting.
		/// If you don't want to buffer up tokens, then this method makes no
		/// sense for you.  Naturally you can't use the rewrite stream feature.
		/// I believe DebugTokenStream can easily be altered to not use
		/// this method, removing the dependency.
		/// </summary>
		IToken Get(int i);
		
		/// <summary>Where is this stream pulling tokens from?  This is not the name, but
		/// the object that provides Token objects.
		/// </summary>
		ITokenSource TokenSource { get; }
		
		/// <summary>Return the text of all tokens from start to stop, inclusive.
		/// If the stream does not buffer all the tokens then it can just
		/// return "" or null;  Users should not access $ruleLabel.text in
		/// an action of course in that case.
		/// </summary>
		string ToString(int start, int stop);
		
		/// <summary>Because the user is not required to use a token with an index stored
		/// in it, we must provide a means for two token objects themselves to
		/// indicate the start/end location.  Most often this will just delegate
		/// to the other toString(int,int).  This is also parallel with
		/// the TreeNodeStream.toString(Object,Object).
		/// </summary>
		string ToString(IToken start, IToken stop);
	}
}