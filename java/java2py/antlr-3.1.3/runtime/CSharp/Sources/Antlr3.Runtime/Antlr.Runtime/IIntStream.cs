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
	
	/// <summary>
    /// A simple stream of integers. This is useful when all we care about is the char
	/// or token type sequence (such as for interpretation).
	/// </summary>
	public interface IIntStream
	{
		void Consume();
		
		/// <summary>
        /// Get int at current input pointer + i ahead (where i=1 is next int)
		/// Negative indexes are allowed.  LA(-1) is previous token (token just matched).
		/// LA(-i) where i is before first token should yield -1, invalid char or EOF.
        /// </summary>
		int LA(int i);
		
		/// <summary>Tell the stream to start buffering if it hasn't already.</summary>
        /// <remarks>
        /// Executing Rewind(Mark()) on a stream should not affect the input position.
		/// The Lexer tracks line/col info as well as input index so its markers are
		/// not pure input indexes.  Same for tree node streams.													*/
        /// </remarks>
        /// <returns>Return a marker that can be passed to 
        /// <see cref="IIntStream.Rewind(int)"/> to return to the current position. 
        /// This could be the current input position, a value return from 
        /// <see cref="IIntStream.Index"/>, or some other marker.</returns>
		int Mark();
		
		/// <summary>
        /// Return the current input symbol index 0..n where n indicates the 
		/// last symbol has been read. The index is the symbol about to be
		/// read not the most recently read symbol.
		/// </summary>
		int Index();
		
        /// <summary>
		/// Resets the stream so that the next call to 
        /// <see cref="IIntStream.Index"/> would  return marker.
        /// </summary>
        /// <remarks>
        /// The marker will usually be <see cref="IIntStream.Index"/> but 
        /// it doesn't have to be.  It's just a marker to indicate what 
        /// state the stream was in.  This is essentially calling 
        /// <see cref="IIntStream.Release"/> and <see cref="IIntStream.Seek"/>.
        /// If there are other markers created after the specified marker, 
        /// this routine must unroll them like a stack.  Assumes the state the 
        /// stream was in when this marker was created.
        /// </remarks>
		void Rewind(int marker);
		
		/// <summary>
		/// Rewind to the input position of the last marker.
		/// </summary>
		/// <remarks>
		/// Used currently only after a cyclic DFA and just before starting 
		/// a sem/syn predicate to get the input position back to the start 
		/// of the decision. Do not "pop" the marker off the state.  Mark(i) 
		/// and Rewind(i) should balance still. It is like invoking 
		/// Rewind(last marker) but it should not "pop" the marker off.  
		/// It's like Seek(last marker's input position).
		/// </remarks>
		void Rewind();

		/// <summary>
        /// You may want to commit to a backtrack but don't want to force the
		/// stream to keep bookkeeping objects around for a marker that is
		/// no longer necessary.  This will have the same behavior as
        /// <see cref="IIntStream.Rewind(int)"/> except it releases resources without 
        /// the backward seek.
		/// </summary>
		/// <remarks>
		/// This must throw away resources for all markers back to the marker
		/// argument. So if you're nested 5 levels of Mark(), and then Release(2)
		/// you have to release resources for depths 2..5.
		/// </remarks>
		void Release(int marker);
		
		/// <summary>
        /// Set the input cursor to the position indicated by index.  This is
		/// normally used to seek ahead in the input stream.
        /// </summary>
        /// <remarks>
        /// No buffering is required to do this unless you know your stream 
        /// will use seek to move backwards such as when backtracking.
		/// 
		/// This is different from rewind in its multi-directional requirement 
        /// and in that its argument is strictly an input cursor (index).
		/// 
		/// For char streams, seeking forward must update the stream state such
		/// as line number.  For seeking backwards, you will be presumably
        /// backtracking using the 
        /// <see cref="IIntStream.Mark"/>/<see cref="IIntStream.Rewind(int)"/> 
        /// mechanism that restores state and so this method does not need to 
        /// update state when seeking backwards.
		/// 
		/// Currently, this method is only used for efficient backtracking using
		/// memoization, but in the future it may be used for incremental parsing.
		/// 
		/// The index is 0..n-1. A seek to position i means that LA(1) will return 
		/// the ith symbol.  So, seeking to 0 means LA(1) will return the first 
		/// element in the stream.
        /// </remarks>
		void Seek(int index);
		
		/// <summary>Returns the size of the entire stream.</summary>
        /// <remarks>
        /// Only makes sense for streams that buffer everything up probably, 
        /// but might be useful to display the entire stream or for testing.
		/// This value includes a single EOF.
        /// </remarks>
        [Obsolete("Please use property Count instead.")]
		int Size();
		
		/// <summary>Returns the size of the entire stream.</summary>
        /// <remarks>
        /// Only makes sense for streams that buffer everything up probably, 
        /// but might be useful to display the entire stream or for testing.
		/// This value includes a single EOF.
        /// </remarks>
        int Count { get; }

		/// <summary>
		/// Where are you getting symbols from?  Normally, implementations will
		/// pass the buck all the way to the lexer who can ask its input stream
		/// for the file name or whatever.
		/// </summary>
		string SourceName {
			get;
		}
	}
}