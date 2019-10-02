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

#if DOTNET1
	public sealed class Token {
		/// <summary>
		/// Instead of a static modifier for Token.
		/// </summary>
		private Token() {
		}
#else
	public static class Token {
#endif
		
		public const int EOR_TOKEN_TYPE = 1;

		/// <summary>imaginary tree navigation type; traverse "get child" link </summary>
		public const int DOWN = 2;
		/// <summary>imaginary tree navigation type; finish with a child list </summary>
		public const int UP = 3;

		public static readonly int MIN_TOKEN_TYPE = UP + 1;

		public static readonly int EOF = (int)CharStreamConstants.EOF;
		public static readonly IToken EOF_TOKEN = new CommonToken(EOF);

		public const int INVALID_TOKEN_TYPE = 0;
		public static readonly IToken INVALID_TOKEN = new CommonToken(INVALID_TOKEN_TYPE);

		/// <summary>
		/// In an action, a lexer rule can set token to this SKIP_TOKEN and ANTLR
		/// will avoid creating a token for this symbol and try to fetch another.
		/// </summary>
		public static readonly IToken SKIP_TOKEN = new CommonToken(INVALID_TOKEN_TYPE);

		/// <summary>
		/// All tokens go to the parser (unless skip() is called in that rule)
		/// on a particular "channel".  The parser tunes to a particular channel
		/// so that whitespace etc... can go to the parser on a "hidden" channel.
		/// </summary>
		public const int DEFAULT_CHANNEL = 0;

		/// <summary>
		/// Anything on different channel than DEFAULT_CHANNEL is not parsed by parser.
		/// </summary>
		public const int HIDDEN_CHANNEL = 99;
	}
}
