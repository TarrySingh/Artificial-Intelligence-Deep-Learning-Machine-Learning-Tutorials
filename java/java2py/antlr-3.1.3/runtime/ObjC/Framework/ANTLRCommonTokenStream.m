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

#import "ANTLRCommonTokenStream.h"


@implementation ANTLRCommonTokenStream

#pragma mark Initialization

- (id) init
{
	if ((self = [self initWithTokenSource:nil])) {
	}
	return self;
}

- (id) initWithTokenSource:(id<ANTLRTokenSource>)theTokenSource
{
	if ((self = [super init])) {
		[self setTokenSource:theTokenSource];
		p = -1;
		discardOffChannelTokens = NO;
		tokens = [[NSMutableArray alloc] initWithCapacity:500];
		channelOverride = [[NSMutableDictionary alloc] init];
		discardSet = [[NSMutableSet alloc] init];
		channel = ANTLRTokenChannelDefault;
	}
	return self;
}

- (void) dealloc
{
	[discardSet release];
	[channelOverride release];
	[tokens release];
	[self setTokenSource:nil];
	[super dealloc];
}

- (void) fillBuffer
{
	int index = 0;
	id<ANTLRToken> token = [tokenSource nextToken];
	while ( token && [token type] != ANTLRTokenTypeEOF ) {
		BOOL discard = NO;
		NSNumber *tokenType = [NSNumber numberWithInt:[token type]];
		if ([discardSet containsObject:tokenType])
		{
			discard = YES;
		} else if ( discardOffChannelTokens && [token channel] != channel ) {
			discard = YES;
		} else {
			NSNumber *channelI = [channelOverride objectForKey:tokenType];
			if (channelI) {
				[token setChannel:[channelI intValue]];
			}
		}
		if ( !discard )	{
			[token setTokenIndex:index];
			[tokens addObject:token];
			index++;
		}
		token = [tokenSource nextToken];
	}
	// leave p pointing at first token on channel
	p = 0;
	p = [self skipOffChannelTokens:p];
}

#pragma mark Accessors

- (id<ANTLRTokenSource>) tokenSource
{
    return tokenSource; 
}

- (void) setTokenSource: (id<ANTLRTokenSource>) aTokenSource
{
    if (tokenSource != aTokenSource) {
        [(id<NSObject>)aTokenSource retain];
        [(id<NSObject>)tokenSource release];
        tokenSource = aTokenSource;
		p = -1;
		channel = ANTLRTokenChannelDefault;
    }
}


#pragma mark Channels & Skipping

- (int) skipOffChannelTokens:(int) i
{
	int n = [tokens count];
	int tmp = i;
	while ( tmp < n && [(id<ANTLRToken>)[tokens objectAtIndex:tmp] channel] != channel ) {
		tmp++;
	}
	return tmp;
}

- (int) skipOffChannelTokensReverse:(int) i
{
	int tmp = i;
	while ( tmp >= 0 && [(id<ANTLRToken>)[tokens objectAtIndex:tmp] channel] != channel ) {
		tmp--;
	}
	return tmp;
}

- (void) setTokenType:(int)ttype toChannel:(int)theChannel
{
	[channelOverride setObject:[NSNumber numberWithInt:theChannel] forKey:[NSNumber numberWithInt:ttype]];
}

- (void) discardTokenType:(int)ttype
{
	[discardSet addObject:[NSNumber numberWithInt:ttype]];
}

- (void) discardOffChannelTokens:(BOOL)flag
{
	discardOffChannelTokens = flag;
}

#pragma mark Token access


- (NSArray *) tokens
{
	if ( p == -1 ) {
		[self fillBuffer];
	}
	return tokens;
}

- (NSArray *) tokensInRange:(NSRange)aRange
{
	return [tokens subarrayWithRange:aRange];
}

- (NSArray *) tokensInRange:(NSRange)aRange inBitSet:(ANTLRBitSet *)aBitSet
{
	unsigned int start = aRange.location;
	unsigned int stop = aRange.location+aRange.length;
	if ( p == -1 ) {
		[self fillBuffer];
	}
	if (stop >= [tokens count]) {
		stop = [tokens count] - 1;
	}
	NSMutableArray *filteredTokens = [[NSMutableArray alloc] init];
	unsigned int i=0;
	for (i = start; i<=stop; i++) {
		id<ANTLRToken> token = [tokens objectAtIndex:i];
		if (aBitSet == nil || [aBitSet isMember:[token type]]) {
			[filteredTokens addObject:token];
		}
	}
	if ([filteredTokens count]) {
		return [filteredTokens autorelease];
	} else {
		[filteredTokens release];
		return nil;
	}
}

- (NSArray *) tokensInRange:(NSRange)aRange withTypes:(NSArray *)tokenTypes
{
	ANTLRBitSet *bits = [[ANTLRBitSet alloc] initWithArrayOfBits:tokenTypes];
	NSArray *returnTokens = [self tokensInRange:aRange inBitSet:bits];
	[bits release];
	return returnTokens;
}

- (NSArray *) tokensInRange:(NSRange)aRange withType:(int)tokenType
{
	ANTLRBitSet *bits = [[ANTLRBitSet alloc] init];
	[bits add:tokenType];
	NSArray *returnTokens = [self tokensInRange:aRange inBitSet:bits];
	[bits release];
	return returnTokens;
}

- (id<ANTLRToken>) tokenAtIndex:(int)i
{
	return [tokens objectAtIndex:i];
}

- (int) count
{
	return [tokens count];
}

#pragma mark Lookahead

- (id<ANTLRToken>) LT:(int)k
{
	if ( p == -1 ) {
		[self fillBuffer];
	}
	if ( k == 0 ) {
		return nil;
	}
	if ( k < 0 ) {
		return [self LB:k];
	}
	if ( (p+k-1) >= (int)[tokens count] ) {
		return [ANTLRCommonToken eofToken];
	}
	int i = p;
	int n = 1;
	while ( n < k ) {
		i = [self skipOffChannelTokens:i+1];
		n++;
	}
	if ( i >= (int)[tokens count] ) {
		return [ANTLRCommonToken eofToken];
	}
	return [tokens objectAtIndex:i];
}

- (id<ANTLRToken>) LB:(int)k
{
	if ( p == -1 ) {
		[self fillBuffer];
	}
	if ( k == 0 ) {
		return nil;
	}
	if ( (p-k)<0 ) {
		return nil;
	}
	if ( (p+k-1) >= (int)[tokens count] ) {
		return [ANTLRCommonToken eofToken];
	}
	int i = p;
	int n = 1;
	while ( n <= k ) {
		i = [self skipOffChannelTokensReverse:i-1];
		n++;
	}
	if ( i-1 < 0 ) {
		return nil;
	}
	return [tokens objectAtIndex:i-1];
}

- (int) LA:(int)k
{
	return [[self LT:k] type];
}

#pragma mark Stream Seeking

- (void) consume
{
	if (p < (int)[tokens count]) {
		p++;
		p = [self skipOffChannelTokens:p];
	}
}

- (int) mark
{
	lastMarker = [self index];
	return lastMarker;
}

- (void) release:(int)marker
{
	// empty
}

- (int) index
{
	return p;
}

- (void) rewind
{
	[self seek:lastMarker];
}

- (void) rewind:(int)marker
{
	[self seek:marker];
}

- (void) seek:(int)index
{
	p = index;
}
#pragma mark Stringvalues

- (NSString *) stringValue
{
	if ( p == -1 ) {
		[self fillBuffer];
	}
	return [self stringValueWithRange:NSMakeRange(0,[self count])];
}

- (NSString *) stringValueWithRange:(NSRange) aRange
{
	if ( p == -1 ) {
		[self fillBuffer];
	}
	NSArray *tokensInRange = [self tokensInRange:aRange];
	NSEnumerator *tokenEnum = [tokensInRange objectEnumerator];
	id<ANTLRToken> token;
	NSMutableString *stringValue = [[NSMutableString alloc] init];
	while ((token = [tokenEnum nextObject])) {
		[stringValue appendString:[token text]];
	}
	return [stringValue autorelease];
}

- (NSString *) stringValueFromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken
{
	if (startToken && stopToken) {
		int start = [startToken tokenIndex];
		int stop = [stopToken tokenIndex];
		return [self stringValueWithRange:NSMakeRange(start,stop-start)];
	}
	return nil;
}


@end
