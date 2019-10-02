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

#import "ANTLRDebugTreeParser.h"


@implementation ANTLRDebugTreeParser

- (id) initWithTreeNodeStream:(id<ANTLRTreeNodeStream>)theStream
{
	return [self initWithTreeNodeStream:theStream debugListener:nil debuggerPort:-1];
}

- (id) initWithTreeNodeStream:(id<ANTLRTreeNodeStream>)theStream
				 debuggerPort:(int)portNumber
{
	return [self initWithTreeNodeStream:theStream debugListener:nil debuggerPort:portNumber];
}

- (id) initWithTreeNodeStream:(id<ANTLRTreeNodeStream>)theStream
				debugListener:(id<ANTLRDebugEventListener>)theDebugListener
				 debuggerPort:(int)portNumber
{
	id<ANTLRDebugEventListener,NSObject> debugger = nil;
	id<ANTLRTreeNodeStream> treeNodeStream = nil;
	if (theDebugListener) {
		debugger = [(id<ANTLRDebugEventListener,NSObject>)theDebugListener retain];
	} else {
		debugger = [[ANTLRDebugEventProxy alloc] initWithGrammarName:[self grammarFileName] debuggerPort:portNumber];
	}
	if (theStream && ![theStream isKindOfClass:[ANTLRDebugTreeNodeStream class]]) {
		treeNodeStream = [[ANTLRDebugTreeNodeStream alloc] initWithTreeNodeStream:theStream debugListener:debugger];
	} else {
		treeNodeStream = [theStream retain];
	}
	self = [super initWithTreeNodeStream:treeNodeStream];
	if (self) {
		[self setDebugListener:debugger];
		[debugger release];
		[treeNodeStream release];
		[debugListener waitForDebuggerConnection];
	}
	return self;
}

- (void) dealloc
{
    [self setDebugListener: nil];
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

#pragma mark -
#pragma mark Overrides

- (void) beginResync
{
	[debugListener beginResync];
}

- (void) endResync
{
	[debugListener endResync];
}
- (void)beginBacktracking:(int)level
{
	[debugListener beginBacktrack:level];
}

- (void)endBacktracking:(int)level wasSuccessful:(BOOL)successful
{
	[debugListener endBacktrack:level wasSuccessful:successful];
}

- (void) recoverFromMismatchedToken:(id<ANTLRIntStream>)inputStream 
						  exception:(NSException *)e 
						  tokenType:(ANTLRTokenType)ttype 
							 follow:(ANTLRBitSet *)follow
{
#warning TODO: recoverFromMismatchedToken in debugger
	[super recoverFromMismatchedToken:inputStream exception:e tokenType:ttype follow:follow];
}

- (void) recoverFromMismatchedSet:(id<ANTLRIntStream>)inputStream
						exception:(NSException *)e
						   follow:(ANTLRBitSet *)follow
{
#warning TODO: recoverFromMismatchedSet in debugger
	[super recoverFromMismatchedSet:inputStream exception:e follow:follow];
}

@end
