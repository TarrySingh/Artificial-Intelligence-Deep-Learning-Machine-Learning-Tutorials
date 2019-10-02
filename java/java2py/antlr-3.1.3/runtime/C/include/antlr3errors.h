#ifndef	_ANTLR3ERRORS_H
#define	_ANTLR3ERRORS_H

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

#define	ANTLR3_SUCCESS	0
#define	ANTLR3_FAIL	1

#define	ANTLR3_TRUE	1
#define	ANTLR3_FALSE	0

/** Indicates end of character stream and is an invalid Unicode code point. */
#define ANTLR3_CHARSTREAM_EOF	0xFFFFFFFF

/** Indicates  memoizing on a rule failed.
 */
#define	MEMO_RULE_FAILED	0xFFFFFFFE
#define	MEMO_RULE_UNKNOWN	0xFFFFFFFF


#define	ANTLR3_ERR_BASE	    0
#define	ANTLR3_ERR_NOMEM    (ANTLR3_ERR_BASE + 1)
#define	ANTLR3_ERR_NOFILE   (ANTLR3_ERR_BASE + 2)
#define	ANTLR3_ERR_HASHDUP  (ANTLR3_ERR_BASE + 3)

#endif	/* _ANTLR3ERRORS_H */
