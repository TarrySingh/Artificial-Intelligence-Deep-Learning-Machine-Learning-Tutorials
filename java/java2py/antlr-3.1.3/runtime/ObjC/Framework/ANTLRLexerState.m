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

#import "ANTLRLexerState.h"


@implementation ANTLRLexerState

- (id) init
{
	self = [super init];
	if (self) {
		[self reset];
	}
	return self;
}

- (void) reset
{
	[super reset];
	[self setToken:nil];
	tokenType = 0;				
	channel = 0;				
	tokenStartLine = 0;		
	tokenCharPositionInLine = 0;
	tokenStartCharIndex = -1;    
	[self setText:nil];
}

- (void) dealloc
{
	[self setText:nil];
	[self setToken:nil];
	[super dealloc];
}

- (id<ANTLRToken>) token
{
	return token;
}
- (void) setToken:(id<ANTLRToken>) theToken
{
	if (theToken != token) {
		[token release];
		token = [theToken retain];
	}
}


- (unsigned int) tokenType
{
	return tokenType;
}
- (void) setTokenType:(unsigned int) theTokenType
{
	tokenType = theTokenType;
}

- (unsigned int) channel
{
	return channel;
}
- (void) setChannel:(unsigned int) theChannel
{
	channel = theChannel;
}

- (unsigned int) tokenStartLine
{
	return tokenStartLine;
}
- (void) setTokenStartLine:(unsigned int) theTokenStartLine
{
	tokenStartLine = theTokenStartLine;
}

- (unsigned int) tokenCharPositionInLine
{
	return tokenCharPositionInLine;
}
- (void) setTokenCharPositionInLine:(unsigned int) theCharPosition
{
	tokenCharPositionInLine = theCharPosition;
}

- (int) tokenStartCharIndex
{
	return tokenStartCharIndex;
}
- (void) setTokenStartCharIndex:(int) theTokenStartCharIndex
{
	tokenStartCharIndex = theTokenStartCharIndex;
}

- (NSString *) text
{
	return text;
}
- (void) setText:(NSString *) theText
{
	if (text != theText) {
		[text release];
		text = [theText retain];
	}
}

@end
