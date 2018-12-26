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


#import <ANTLR/ANTLRIntStream.h>
#import <ANTLR/ANTLRToken.h>

@protocol ANTLRTokenStream < ANTLRIntStream >

// Get Token at current input pointer + i ahead where i=1 is next Token.
// i<0 indicates tokens in the past.  So -1 is previous token and -2 is
// two tokens ago. LT:0 is undefined.  For i>=n, return Token.EOFToken.
// Return null for LT:0 and any index that results in an absolute address
// that is negative.

- (id<ANTLRToken>) LT:(int) i;

- (id<ANTLRToken>) tokenAtIndex:(unsigned int) i;

- (id) tokenSource;

- (NSString *) stringValue;
- (NSString *) stringValueWithRange:(NSRange) aRange;
- (NSString *) stringValueFromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken;


@end
