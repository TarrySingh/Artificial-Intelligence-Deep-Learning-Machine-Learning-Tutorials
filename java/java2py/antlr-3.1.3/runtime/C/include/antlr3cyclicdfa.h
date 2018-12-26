/// Definition of a cyclic dfa structure such that it can be
/// initialized at compile time and have only a single
/// runtime function that can deal with all cyclic dfa
/// structures and show Java how it is done ;-)
///
#ifndef	ANTLR3_CYCLICDFA_H
#define	ANTLR3_CYCLICDFA_H

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

#include    <antlr3baserecognizer.h>
#include    <antlr3intstream.h>

#ifdef __cplusplus
extern "C" {

// If this header file is included as part of a generated recognizer that
// is being compiled as if it were C++, and this is Windows, then the const elements
// of the structure cause the C++ compiler to (rightly) point out that
// there can be no instantiation of the structure because it needs a constructor
// that can initialize the data, however these structures are not
// useful for C++ as they are pre-generated and static in the recognizer.
// So, we turn off those warnings, which are only at /W4 anyway.
//
#ifdef ANTLR3_WINDOWS
#pragma warning	(push)
#pragma warning (disable : 4510)
#pragma warning (disable : 4512)
#pragma warning (disable : 4610)
#endif
#endif

typedef struct ANTLR3_CYCLIC_DFA_struct
{
    /// Decision number that a particular static structure
    ///  represents.
    ///
    const ANTLR3_INT32		decisionNumber;

    /// What this decision represents
    ///
    const pANTLR3_UCHAR		description;

    ANTLR3_INT32			(*specialStateTransition)   (void * ctx, pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM is, struct ANTLR3_CYCLIC_DFA_struct * dfa, ANTLR3_INT32 s);

    ANTLR3_INT32			(*specialTransition)	    (void * ctx, pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM is, struct ANTLR3_CYCLIC_DFA_struct * dfa, ANTLR3_INT32 s);

    ANTLR3_INT32			(*predict)					(void * ctx, pANTLR3_BASE_RECOGNIZER recognizer, pANTLR3_INT_STREAM is, struct ANTLR3_CYCLIC_DFA_struct * dfa);

    const ANTLR3_INT32		    * const eot;
    const ANTLR3_INT32		    * const eof;
    const ANTLR3_INT32		    * const min;
    const ANTLR3_INT32		    * const max;
    const ANTLR3_INT32		    * const accept;
    const ANTLR3_INT32		    * const special;
    const ANTLR3_INT32			* const * const transition;

}
    ANTLR3_CYCLIC_DFA;

typedef ANTLR3_INT32		(*CDFA_SPECIAL_FUNC)   (void * , pANTLR3_BASE_RECOGNIZER , pANTLR3_INT_STREAM , struct ANTLR3_CYCLIC_DFA_struct * , ANTLR3_INT32);

#ifdef __cplusplus
}
#ifdef ANTLR3_WINDOWS
#pragma warning	(pop)
#endif
#endif

#endif
