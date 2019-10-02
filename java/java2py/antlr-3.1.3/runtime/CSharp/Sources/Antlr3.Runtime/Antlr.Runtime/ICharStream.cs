/*
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
	
	public enum CharStreamConstants
    {
		EOF = - 1,
	}

    /// <summary>A source of characters for an ANTLR lexer </summary>
    public interface ICharStream : IIntStream
	{
        /// <summary>
        /// The current line in the character stream (ANTLR tracks the 
        /// line information automatically. To support rewinding character 
        /// streams, we are able to [re-]set the line.
        /// </summary>
        int Line
		{
            get;
            set;
		}

        /// <summary>
        /// The index of the character relative to the beginning of the 
        /// line (0..n-1). To support rewinding character streams, we are 
        /// able to [re-]set the character position.
        /// </summary>
		int CharPositionInLine
		{
			get;
			set;
		}

        /// <summary>
        /// Get the ith character of lookahead.  This is usually the same as
        /// LA(i).  This will be used for labels in the generated lexer code.
        /// I'd prefer to return a char here type-wise, but it's probably 
        /// better to be 32-bit clean and be consistent with LA.
        /// </summary>
        int LT(int i);
		
		/// <summary>
        /// This primarily a useful interface for action code (just make sure 
        /// actions don't use this on streams that don't support it).
        /// For infinite streams, you don't need this.
		/// </summary>
		string Substring(int start, int stop);
	}
}