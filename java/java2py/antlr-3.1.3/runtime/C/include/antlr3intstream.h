/** \file
 * Defines the the class interface for an antlr3 INTSTREAM.
 * 
 * Certain functionality (such as DFAs for instance) abstract the stream of tokens
 * or characters in to a steam of integers. Hence this structure should be included
 * in any stream that is able to provide the output as a stream of integers (which is anything
 * basically.
 *
 * There are no specific implementations of the methods in this interface in general. Though
 * for purposes of casting and so on, it may be necesssary to implement a function with
 * the signature in this interface which abstracts the base immplementation. In essence though
 * the base stream provides a pointer to this interface, within which it installs its
 * normal match() functions and so on. Interaces such as DFA are then passed the pANTLR3_INT_STREAM
 * and can treat any input as an int stream. 
 *
 * For instance, a lexer implements a pANTLR3_BASE_RECOGNIZER, within which there is a pANTLR3_INT_STREAM.
 * However, a pANTLR3_INPUT_STREAM also provides a pANTLR3_INT_STREAM, which it has constructed from
 * it's normal interface when it was created. This is then pointed at by the pANTLR_BASE_RECOGNIZER
 * when it is intialized with a pANTLR3_INPUT_STREAM.
 *
 * Similarly if a pANTLR3_BASE_RECOGNIZER is initialized with a pANTLR3_TOKEN_STREAM, then the 
 * pANTLR3_INT_STREAM is taken from the pANTLR3_TOKEN_STREAM. 
 *
 * If a pANTLR3_BASE_RECOGNIZER is initialized with a pANTLR3_TREENODE_STREAM, then guess where
 * the pANTLR3_INT_STREAM comes from?
 *
 * Note that because the context pointer points to the actual interface structure that is providing
 * the ANTLR3_INT_STREAM it is defined as a (void *) in this interface. There is no direct implementation
 * of an ANTLR3_INT_STREAM (unless someone did not understand what I was doing here =;?P
 */
#ifndef	_ANTLR3_INTSTREAM_H
#define	_ANTLR3_INTSTREAM_H

// [The "BSD licence"]
// Copyright (c) 2005-2009 Jim Idle, Temporal Wave LLC
// http://www.temporal-wave.com
// http://www.linkedin.com/in/jimidle
//
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

#include    <antlr3defs.h>
#include    <antlr3commontoken.h>

/** Type indicator for a character stream
 * \remark if a custom stream is created but it can be treated as
 * a char stream, then you may OR in this value to your type indicator
 */
#define	ANTLR3_CHARSTREAM	0x0001

/** Type indicator for a Token stream
 * \remark if a custom stream is created but it can be treated as
 * a token stream, then you may OR in this value to your type indicator
 */
#define	ANTLR3_TOKENSTREAM	0x0002

/** Type indicator for a common tree node stream
 * \remark if a custom stream is created but it can be treated as
 * a common tree node stream, then you may OR in this value to your type indicator
 */
#define	ANTLR3_COMMONTREENODE	0x0004

/** Type mask for input stream so we can switch in the above types
 *  \remark DO NOT USE 0x0000 as a stream type!
 */
#define	ANTLR3_INPUT_MASK	0x0007

#ifdef __cplusplus
extern "C" {
#endif

typedef	struct ANTLR3_INT_STREAM_struct
{
    /** Input stream type indicator. Sometimes useful for error reporting etc.
     */
    ANTLR3_UINT32	    type;

    /** Potentially useful in error reporting and so on, this string is
     *  an identification of the input source. It may be NULL, so anything
     *  attempting to access it needs to check this and substitute a sensible
     *  default.
     */
    pANTLR3_STRING	      streamName;

    /** Pointer to the super structure that contains this interface. This
     *  will usually be a token stream or a tree stream.
     */
    void		    * super;

    /** Last marker position allocated
     */
    ANTLR3_MARKER	    lastMarker;

	// Return a string that identifies the input source
	//
	pANTLR3_STRING		(*getSourceName)	(struct ANTLR3_INT_STREAM_struct * intStream);

    /** Consume the next 'ANTR3_UINT32' in the stream
     */
    void		    (*consume)	    (struct ANTLR3_INT_STREAM_struct * intStream);

    /** Get ANTLR3_UINT32 at current input pointer + i ahead where i=1 is next ANTLR3_UINT32 
     */
    ANTLR3_UINT32	    (*_LA)	    (struct ANTLR3_INT_STREAM_struct * intStream, ANTLR3_INT32 i);

    /** Tell the stream to start buffering if it hasn't already.  Return
     *  current input position, index(), or some other marker so that
     *  when passed to rewind() you get back to the same spot.
     *  rewind(mark()) should not affect the input cursor.
     */
    ANTLR3_MARKER	    (*mark)	    (struct ANTLR3_INT_STREAM_struct * intStream);
    
    /** Return the current input symbol index 0..n where n indicates the
     *  last symbol has been read.
     */
    ANTLR3_MARKER	    (*index)	    (struct ANTLR3_INT_STREAM_struct * intStream);

    /** Reset the stream so that next call to index would return marker.
     *  The marker will usually be index() but it doesn't have to be.  It's
     *  just a marker to indicate what state the stream was in.  This is
     *  essentially calling release() and seek().  If there are markers
     *  created after this marker argument, this routine must unroll them
     *  like a stack.  Assume the state the stream was in when this marker
     *  was created.
     */
    void		    (*rewind)	    (struct ANTLR3_INT_STREAM_struct * intStream, ANTLR3_MARKER marker);

    /** Reset the stream to the last marker position, witouh destryoing the
     *  last marker position.
     */
    void		    (*rewindLast)   (struct ANTLR3_INT_STREAM_struct * intStream);

    /** You may want to commit to a backtrack but don't want to force the
     *  stream to keep bookkeeping objects around for a marker that is
     *  no longer necessary.  This will have the same behavior as
     *  rewind() except it releases resources without the backward seek.
     */
    void		    (*release)	    (struct ANTLR3_INT_STREAM_struct * intStream, ANTLR3_MARKER mark);

    /** Set the input cursor to the position indicated by index.  This is
     *  normally used to seek ahead in the input stream.  No buffering is
     *  required to do this unless you know your stream will use seek to
     *  move backwards such as when backtracking.
     *
     *  This is different from rewind in its multi-directional
     *  requirement and in that its argument is strictly an input cursor (index).
     *
     *  For char streams, seeking forward must update the stream state such
     *  as line number.  For seeking backwards, you will be presumably
     *  backtracking using the mark/rewind mechanism that restores state and
     *  so this method does not need to update state when seeking backwards.
     *
     *  Currently, this method is only used for efficient backtracking, but
     *  in the future it may be used for incremental parsing.
     */
    void		    (*seek)	    (struct ANTLR3_INT_STREAM_struct * intStream, ANTLR3_MARKER index);

    /** Only makes sense for streams that buffer everything up probably, but
     *  might be useful to display the entire stream or for testing.
     */
    ANTLR3_UINT32	    (*size)	    (struct ANTLR3_INT_STREAM_struct * intStream);

    /** Because the indirect call, though small in individual cases can
     *  mount up if there are thousands of tokens (very large input streams), callers
     *  of size can optionally use this cached size field.
     */
    ANTLR3_UINT32	    cachedSize;

    /** Frees any resources that were allocated for the implementation of this
     *  interface. Usually this is just releasing the memory allocated
     *  for the structure itself, but it may of course do anything it need to
     *  so long as it does not stamp on anything else.
     */
    void		    (*free)	    (struct ANTLR3_INT_STREAM_struct * stream);

}
    ANTLR3_INT_STREAM;

#ifdef __cplusplus
}
#endif

#endif

