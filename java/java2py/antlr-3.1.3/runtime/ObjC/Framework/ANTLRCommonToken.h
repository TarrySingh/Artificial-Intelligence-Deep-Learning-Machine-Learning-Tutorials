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
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRCharStream.h>

@interface ANTLRCommonToken : NSObject < ANTLRToken > {
	NSString *text;
	
	int type;
	// information about the Token's position in the input stream
	unsigned int line;
	unsigned int charPositionInLine;
	unsigned int channel;
	
	// the actual input stream this token was found in
	id<ANTLRCharStream> input;
	// indices into the CharStream to avoid copying the text
	// can manually override the text by using -setText:
	unsigned int start;
	unsigned int stop;
	// this token's position in the TokenStream
	unsigned int index;
}

// designated initializer. This is used as the default way to initialize a Token in the generated code.
- (ANTLRCommonToken *) initWithInput:(id<ANTLRCharStream>)anInput tokenType:(int)aTType channel:(int)aChannel start:(int)theStart stop:(int)theStop;
- (ANTLRCommonToken *) initWithToken:(ANTLRCommonToken *)aToken;

- (id<ANTLRCharStream>) input;
- (void) setInput: (id<ANTLRCharStream>) anInput;

- (unsigned int) start;
- (void) setStart: (unsigned int) aStart;

- (unsigned int) stop;
- (void) setStop: (unsigned int) aStop;

// the index of this Token into the TokenStream
- (unsigned int) tokenIndex;
- (void) setTokenIndex: (unsigned int) aTokenIndex;

// conform to NSCopying
- (id) copyWithZone:(NSZone *)theZone;

@end
