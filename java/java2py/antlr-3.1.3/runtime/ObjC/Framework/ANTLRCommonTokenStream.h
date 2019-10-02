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
#import <ANTLR/ANTLRTokenStream.h>
#import <ANTLR/ANTLRToken.h>
#import <ANTLR/ANTLRCommonToken.h>
#import <ANTLR/ANTLRTokenSource.h>
#import <ANTLR/ANTLRBitSet.h>

@interface ANTLRCommonTokenStream : NSObject < ANTLRTokenStream > {
	id<ANTLRTokenSource> tokenSource;
	NSMutableArray *tokens;

	NSMutableDictionary *channelOverride;
	NSMutableSet *discardSet;
	unsigned int channel;
	BOOL discardOffChannelTokens;
	int lastMarker;
    int p;
}

- (id) initWithTokenSource:(id<ANTLRTokenSource>)theTokenSource;

- (id<ANTLRTokenSource>) tokenSource;
- (void) setTokenSource: (id<ANTLRTokenSource>) aTokenSource;

- (void) fillBuffer;
- (void) consume;

- (int) skipOffChannelTokens:(int) i;
- (int) skipOffChannelTokensReverse:(int) i;

- (void) setTokenType:(int)ttype toChannel:(int)channel;
- (void) discardTokenType:(int)ttype;
- (void) discardOffChannelTokens:(BOOL)flag;

- (NSArray *) tokens;
- (NSArray *) tokensInRange:(NSRange)aRange;
- (NSArray *) tokensInRange:(NSRange)aRange inBitSet:(ANTLRBitSet *)aBitSet;
- (NSArray *) tokensInRange:(NSRange)aRange withTypes:(NSArray *)tokenTypes;
- (NSArray *) tokensInRange:(NSRange)aRange withType:(int)tokenType;

- (id<ANTLRToken>) LT:(int)k;
- (id<ANTLRToken>) LB:(int)k;
- (int) LA:(int)k;

- (id<ANTLRToken>) tokenAtIndex:(int)i;

- (int) mark;
- (void) release:(int)marker;

- (int) count;
- (int) index;
- (void) rewind;
- (void) rewind:(int)marker;
- (void) seek:(int)index;

- (NSString *) stringValue;
- (NSString *) stringValueWithRange:(NSRange) aRange;
- (NSString *) stringValueFromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken;

@end
