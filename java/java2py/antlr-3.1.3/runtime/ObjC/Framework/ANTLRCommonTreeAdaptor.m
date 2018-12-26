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

#import "ANTLRCommonTreeAdaptor.h"

@implementation ANTLRCommonTreeAdaptor

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>) payload
{
    // Had to cast to id<ANTLRTree> here, because GCC is dumb.
	return (id<ANTLRTree>)[[ANTLRCommonTree alloc] initWithToken:(ANTLRCommonToken *)payload];
}

- (id<ANTLRToken>) newTokenWithToken:(id<ANTLRToken>)fromToken
{
	return [[ANTLRCommonToken alloc] initWithToken:(ANTLRCommonToken *)fromToken];
}

- (id<ANTLRToken>) newTokenWithTokenType:(int)tokenType text:(NSString *)tokenText
{
	ANTLRCommonToken *newToken = [[ANTLRCommonToken alloc] init];
	[newToken setType:tokenType];
	[newToken setText:tokenText];
	return newToken;
}

- (void) setBoundariesForTree:(id<ANTLRTree>)aTree fromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken
{
	ANTLRCommonTree *tmpTree = (ANTLRCommonTree *)aTree;
	[tmpTree setStartIndex:[startToken tokenIndex]];
	[tmpTree setStopIndex:[stopToken tokenIndex]];
		
}

- (int) tokenStartIndexForTree:(id<ANTLRTree>)aTree
{
	return [(ANTLRCommonTree *)aTree startIndex];
}

- (int) tokenStopIndexForTree:(id<ANTLRTree>)aTree
{
	return [(ANTLRCommonTree *)aTree stopIndex];
}

@end
