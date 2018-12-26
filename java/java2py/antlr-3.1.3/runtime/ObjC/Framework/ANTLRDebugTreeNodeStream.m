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

#import "ANTLRDebugTreeNodeStream.h"


@implementation ANTLRDebugTreeNodeStream

- (id) initWithTreeNodeStream:(id<ANTLRTreeNodeStream>)theStream debugListener:(id<ANTLRDebugEventListener>)debugger
{
	self = [super init];
	if (self) {
		[self setDebugListener:debugger];
		[self setTreeAdaptor:[theStream treeAdaptor]];
		[self setTreeNodeStream:theStream];
	}
	return self;
}

- (void) dealloc
{
    [self setDebugListener: nil];
    [self setTreeAdaptor: nil];
    [self setTreeNodeStream: nil];
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


- (id<ANTLRTreeAdaptor>) treeAdaptor
{
    return treeAdaptor; 
}

- (void) setTreeAdaptor: (id<ANTLRTreeAdaptor>) aTreeAdaptor
{
    if (treeAdaptor != aTreeAdaptor) {
        [(id<ANTLRTreeAdaptor,NSObject>)aTreeAdaptor retain];
        [(id<ANTLRTreeAdaptor,NSObject>)treeAdaptor release];
        treeAdaptor = aTreeAdaptor;
    }
}


- (id<ANTLRTreeNodeStream>) treeNodeStream
{
    return treeNodeStream; 
}

- (void) setTreeNodeStream: (id<ANTLRTreeNodeStream>) aTreeNodeStream
{
    if (treeNodeStream != aTreeNodeStream) {
        [(id<ANTLRTreeNodeStream,NSObject>)aTreeNodeStream retain];
        [(id<ANTLRTreeNodeStream,NSObject>)treeNodeStream release];
        treeNodeStream = aTreeNodeStream;
    }
}


#pragma mark ANTLRTreeNodeStream conformance

- (id) LT:(int)k
{
	id node = [treeNodeStream LT:k];
	unsigned hash = [treeAdaptor uniqueIdForTree:node];
	NSString *text = [treeAdaptor textForNode:node];
	int type = [treeAdaptor tokenTypeForNode:node];
	[debugListener LT:k foundNode:hash ofType:type text:text];
	return node;
}

- (void) setUsesUniqueNavigationNodes:(BOOL)flag
{
	[treeNodeStream setUsesUniqueNavigationNodes:flag];
}

#pragma mark ANTLRIntStream conformance
- (void) consume
{
	id node = [treeNodeStream LT:1];
	[treeNodeStream consume];
	unsigned hash = [treeAdaptor uniqueIdForTree:node];
	NSString *text = [treeAdaptor textForNode:node];
	int type = [treeAdaptor tokenTypeForNode:node];
	[debugListener consumeNode:hash ofType:type text:text];
}

- (int) LA:(unsigned int) i
{
	id<ANTLRTree> node = [self LT:1];
	return [node tokenType];
}

- (unsigned int) mark
{
	unsigned lastMarker = [treeNodeStream mark];
	[debugListener mark:lastMarker];
	return lastMarker;
}

- (unsigned int) index
{
	return [treeNodeStream index];
}

- (void) rewind:(unsigned int) marker
{
	[treeNodeStream rewind:marker];
	[debugListener rewind:marker];
}

- (void) rewind
{
	[treeNodeStream rewind];
	[debugListener rewind];
}

- (void) release:(unsigned int) marker
{
	[treeNodeStream release:marker];
}

- (void) seek:(unsigned int) index
{
	[treeNodeStream seek:index];
	// todo: seek missing in debug protocol
}

- (unsigned int) count
{
	return [treeNodeStream count];
}




@end
