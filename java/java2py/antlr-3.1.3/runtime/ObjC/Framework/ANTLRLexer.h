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


#import <Cocoa/Cocoa.h>
#import <ANTLR/ANTLRTokenSource.h>
#import <ANTLR/ANTLRBaseRecognizer.h>
#import <ANTLR/ANTLRCharStream.h>
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRCommonToken.h>
#import <ANTLR/ANTLRRecognitionException.h>
#import <ANTLR/ANTLRMismatchedTokenException.h>
#import <ANTLR/ANTLRMismatchedRangeException.h>
#import <ANTLR/ANTLRLexerState.h>

@interface ANTLRLexer : ANTLRBaseRecognizer <ANTLRTokenSource> {
	id<ANTLRCharStream> input;      ///< The character stream we pull tokens out of.
	unsigned int ruleNestingLevel;
}

#pragma mark Initializer
- (id) initWithCharStream:(id<ANTLRCharStream>)anInput;

- (ANTLRLexerState *) state;

#pragma mark Tokens
- (id<ANTLRToken>) token;
- (void) setToken: (id<ANTLRToken>) aToken;
- (id<ANTLRToken>) nextToken;
- (void) mTokens;		// abstract, defined in generated sources
- (id<ANTLRCharStream>) input;
- (void) setInput:(id<ANTLRCharStream>)aCharStream;

- (void) emit;
- (void) emit:(id<ANTLRToken>)aToken;

#pragma mark Matching
- (void) matchString:(NSString *)aString;
- (void) matchAny;
- (void) matchChar:(unichar) aChar;
- (void) matchRangeFromChar:(unichar)fromChar to:(unichar)toChar;

#pragma mark Informational
- (unsigned int) line;
- (unsigned int) charPositionInLine;
- (int) charIndex;
- (NSString *) text;
- (void) setText:(NSString *) theText;

// error handling
- (void) reportError:(ANTLRRecognitionException *)e;
- (void) recover:(ANTLRRecognitionException *)e;

@end
