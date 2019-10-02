// [The "BSD licence"]
// Copyright (c) 2006-2007 Kay Roepke
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@protocol ANTLRIntStream < NSObject >

- (void) consume;

// Get unichar at current input pointer + i ahead where i=1 is next character as int for including ANTLRCharStreamEOF (-1) in the data range
- (int) LA:(int) i;

// Tell the stream to start buffering if it hasn't already.  Return
// current input position, index(), or some other marker so that
// when passed to rewind() you get back to the same spot.
// rewind(mark()) should not affect the input cursor.
// TODO: problem in that lexer stream returns not index but some marker 

- (int) mark;

// Return the current input symbol index 0..n where n indicates the
// last symbol has been read.

- (int) index;

// Reset the stream so that next call to index would return marker.
// The marker will usually be -index but it doesn't have to be.  It's
// just a marker to indicate what state the stream was in.  This is
// essentially calling -release: and -seek:.  If there are markers
// created after this marker argument, this routine must unroll them
// like a stack.  Assume the state the stream was in when this marker
// was created.

- (void) rewind;
- (void) rewind:(int) marker;

// You may want to commit to a backtrack but don't want to force the
// stream to keep bookkeeping objects around for a marker that is
// no longer necessary.  This will have the same behavior as
// rewind() except it releases resources without the backward seek.

- (void) release:(int) marker;

// Set the input cursor to the position indicated by index.  This is
// normally used to seek ahead in the input stream.  No buffering is
// required to do this unless you know your stream will use seek to
// move backwards such as when backtracking.
// This is different from rewind in its multi-directional
// requirement and in that its argument is strictly an input cursor (index).
//
// For char streams, seeking forward must update the stream state such
// as line number.  For seeking backwards, you will be presumably
// backtracking using the mark/rewind mechanism that restores state and
// so this method does not need to update state when seeking backwards.
//
// Currently, this method is only used for efficient backtracking, but
// in the future it may be used for incremental parsing.

- (void) seek:(int) index;

// Only makes sense for streams that buffer everything up probably, but
// might be useful to display the entire stream or for testing.

- (unsigned int) count;


@end
