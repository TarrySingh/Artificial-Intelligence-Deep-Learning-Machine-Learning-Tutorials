/// \file
/// Implementation of superclass elements of an ANTLR3 int stream.
/// The only methods required are an allocator and a destructor.
/// \addtogroup pANTLR3_INT_STREAM
/// @{

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

#include    <antlr3intstream.h>

static	void	freeStream    (pANTLR3_INT_STREAM stream);

ANTLR3_API pANTLR3_INT_STREAM
antlr3IntStreamNew()
{
	pANTLR3_INT_STREAM	stream;

	// Allocate memory
	//
	stream  = (pANTLR3_INT_STREAM) ANTLR3_CALLOC(1, sizeof(ANTLR3_INT_STREAM));

	if	(stream == NULL)
	{
		return	NULL;
	}

	stream->free    =  freeStream;

	return stream;
}

static	void	
freeStream    (pANTLR3_INT_STREAM stream)
{
	ANTLR3_FREE(stream);
}

/// @}
///
