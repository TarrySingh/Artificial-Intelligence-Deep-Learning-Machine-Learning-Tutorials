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

#import "ANTLRDebugTokenStream.h"


@implementation ANTLRDebugTokenStream


- (id) initWithTokenStream:(id<ANTLRTokenStream>)theStream debugListener:(id<ANTLRDebugEventListener>)debugger
{
	self = [super init];
	if (self) {
		[self setDebugListener:debugger];
		[self setTokenStream:theStream];
		[[self tokenStream] LT:1];	// force reading first on-channel token
		initialStreamState = YES;
	}
	return self;
}

- (void) dealloc
{
    [self setDebugListener: nil];
    [self setTokenStream: nil];
    [super dealloc];
}


- (id<ANTLRDebugEventListener>) debugListener
{
    return debugListener; 
}

- (void) setDebugListener: (id<ANTLRDebugEventListener>) aDebugListener
{
    if (debugListener != aDebugListener) {
        [(id<ANTLRDebugEventListener,NSObject>)aDebugListener retain];
        [(id<ANTLRDebugEventListener,NSObject>)debugListener release];
        debugListener = aDebugListener;
    }
}

- (id<ANTLRTokenStream>) tokenStream
{
    return tokenStream; 
}

- (void) setTokenStream: (id<ANTLRTokenStream>) aTokenStream
{
    if (tokenStream != aTokenStream) {
        [aTokenStream retain];
        [tokenStream release];
        tokenStream = aTokenStream;
    }
}

- (void) consumeInitialHiddenTokens
{
	int firstIdx = [tokenStream index];
	for (int i = 0; i<firstIdx; i++)
		[debugListener consumeHiddenToken:[tokenStream tokenAtIndex:i]];
	initialStreamState = NO;
}

#pragma mark -
#pragma mark Proxy implementation

// anything else that hasn't some debugger event assicioated with it, is simply
// forwarded to the actual token stream
- (void) forwardInvocation:(NSInvocation *)anInvocation
{
	[anInvocation invokeWithTarget:[self tokenStream]];
}

- (void) consume
{
	if ( initialStreamState )
		[self consumeInitialHiddenTokens];
	int a = [tokenStream index];
	id<ANTLRToken> token = [tokenStream LT:1];
	[tokenStream consume];
	int b = [tokenStream index];
	[debugListener consumeToken:token];
	if (b > a+1) // must have consumed hidden tokens
		for (int i = a+1; i < b; i++)
			[debugListener consumeHiddenToken:[tokenStream tokenAtIndex:i]];
}

- (int) mark
{
	int lastMarker = [tokenStream mark];
	[debugListener mark:lastMarker];
	return lastMarker;
}

- (void) rewind
{
	[debugListener rewind];
	[tokenStream rewind];
}

- (void) rewind:(int)marker
{
	[debugListener rewind:marker];
	[tokenStream rewind:marker];
}

- (id<ANTLRToken>) LT:(int)k
{
	if ( initialStreamState )
		[self consumeInitialHiddenTokens];
	[debugListener LT:k foundToken:[tokenStream LT:k]];
	return [tokenStream LT:k];
}

- (int) LA:(int)k
{
	if ( initialStreamState )
		[self consumeInitialHiddenTokens];
	[debugListener LT:k foundToken:[tokenStream LT:k]];
	return [tokenStream LA:k];
}

@end
