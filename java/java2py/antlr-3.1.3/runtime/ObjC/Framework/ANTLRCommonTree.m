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

#import "ANTLRCommonTree.h"


@implementation ANTLRCommonTree

- (id) init
{
	if ((self = [super init]) != nil) {
		startIndex = -1;
		stopIndex = -1;
	}
	return self;
}

- (id<ANTLRTree>) initWithTreeNode:(ANTLRCommonTree *)aNode
{
	if ([self init]) {
		[self setToken:[aNode token]];
	}
	return self;
}

- (id<ANTLRTree>) initWithToken:(ANTLRCommonToken *)aToken
{
	if ([self init]) {
		[self setToken:aToken];
	}
	return self;
}

- (id<ANTLRTree>) initWithTokenType:(int)aTokenType
{
	if ([self init]) {
		ANTLRCommonToken *tmpToken = [[ANTLRCommonToken alloc] init];
		[tmpToken setType:aTokenType];
		[self setToken:tmpToken];
		[tmpToken release];
	}
	return self;
}

- (void) dealloc
{
	[self setToken:nil];
	[super dealloc];
}

- (id<ANTLRTree>) copyWithZone:(NSZone *)aZone
{
	return [[ANTLRCommonTree allocWithZone:aZone] initWithTreeNode:self];
}

- (BOOL) isEmpty
{
	return token == nil;
}

- (ANTLRCommonToken *) token
{
	return token;
}

- (void) setToken:(ANTLRCommonToken *) aToken
{
	if (token != aToken) {
		[aToken retain];
		[token release];
		token = aToken;
	}
}

- (int) tokenType
{
	if (token)
		return [token type];
	return ANTLRTokenTypeInvalid;
}

- (NSString *) text
{
	if (token)
		return [token text];
	return nil;
}

- (unsigned int) line
{
	if (token)
		return [token line];
	return 0;
}

- (unsigned int) charPositionInLine
{
	if (token)
		return [token charPositionInLine];
	return 0;
}

- (int) startIndex
{
    return startIndex;
}

- (void) setStartIndex: (int) aStartIndex
{
    startIndex = aStartIndex;
}

- (int) stopIndex
{
    return stopIndex;
}

- (void) setStopIndex: (int) aStopIndex
{
    stopIndex = aStopIndex;
}

- (NSString *) treeDescription
{
	if (children) {
		NSMutableString *desc = [NSMutableString stringWithString:@"( ^"];
		[desc appendString:[self description]];
		unsigned int childIdx;
		for (childIdx = 0; childIdx < [children count]; childIdx++) {
			[desc appendFormat:@" %@", [[children objectAtIndex:childIdx] treeDescription]];
		}
		[desc appendString:@" )"];
		return desc;
	} else {
		return [self description];
	}
}

- (NSString *) description
{
	if (token)
		return [NSString stringWithFormat:@"\"%@\"", [token text]];
	return @"nil";
}


@end
