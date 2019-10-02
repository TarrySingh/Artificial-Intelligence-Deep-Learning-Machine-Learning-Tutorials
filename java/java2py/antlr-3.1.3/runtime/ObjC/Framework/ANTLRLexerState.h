// [The "BSD licence"]
// Copyright (c) 2007 Kay Roepke
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
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRBaseRecognizerState.h>

@interface ANTLRLexerState : ANTLRBaseRecognizerState {
	id<ANTLRToken> token;					///< The current token that will be emitted next.
    
    unsigned int tokenType;					///< The type of the current token.
    unsigned int channel;					///< The token channel number to be used for the current token.
    unsigned int tokenStartLine;			///< The line number of the first character of the current token appeared in.
    unsigned int tokenCharPositionInLine;	///< The character index of the first character of the current token within the current line.
	int tokenStartCharIndex;				///< The index of the first character of the current token. Default is -1 for an undefined value.
    NSString *text;							///< The text for the current token to be emitted next. If nil, we just refer to the start and stop indices into the character stream.
}

- (void) reset;

- (id<ANTLRToken>) token;
- (void) setToken:(id<ANTLRToken>) theToken;

- (unsigned int) tokenType;
- (void) setTokenType:(unsigned int) theTokenType;

- (unsigned int) channel;
- (void) setChannel:(unsigned int) theChannel;

- (unsigned int) tokenStartLine;
- (void) setTokenStartLine:(unsigned int) theTokenStartLine;

- (unsigned int) tokenCharPositionInLine;
- (void) setTokenCharPositionInLine:(unsigned int) theCharPosition;

- (int) tokenStartCharIndex;
- (void) setTokenStartCharIndex:(int) theTokenStartCharIndex;

- (NSString *) text;
- (void) setText:(NSString *) theText;

@end
