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

#import "ANTLRUnbufferedCommonTreeNodeStreamState.h"


@implementation ANTLRUnbufferedCommonTreeNodeStreamState

- (id) init
{
	if ((self = [super init]) != nil) {
		lookahead = [[NSMutableArray alloc] init];
	}
	return self;
}

- (void) dealloc
{
	[self setLookahead:nil];
	[self setCurrentNode:nil];
	[self setPreviousNode:nil];
	[super dealloc];
}

- (ANTLRCommonTree *) currentNode
{
    return currentNode; 
}

- (void) setCurrentNode: (ANTLRCommonTree *) aCurrentNode
{
    if (currentNode != aCurrentNode) {
        [aCurrentNode retain];
        [currentNode release];
        currentNode = aCurrentNode;
    }
}

- (ANTLRCommonTree *) previousNode
{
    return previousNode; 
}

- (void) setPreviousNode: (ANTLRCommonTree *) aPreviousNode
{
    if (previousNode != aPreviousNode) {
        [aPreviousNode retain];
        [previousNode release];
        previousNode = aPreviousNode;
    }
}

- (int) currentChildIndex
{
    return currentChildIndex;
}

- (void) setCurrentChildIndex: (int) aCurrentChildIndex
{
    currentChildIndex = aCurrentChildIndex;
}

- (int) absoluteNodeIndex
{
    return absoluteNodeIndex;
}

- (void) setAbsoluteNodeIndex: (int) anAbsoluteNodeIndex
{
    absoluteNodeIndex = anAbsoluteNodeIndex;
}

- (unsigned int) nodeStackSize
{
    return nodeStackSize;
}

- (void) setNodeStackSize: (unsigned int) aNodeStackSize
{
    nodeStackSize = aNodeStackSize;
}

- (unsigned int) indexStackSize
{
    return indexStackSize;
}

- (void) setIndexStackSize: (unsigned int) anIndexStackSize
{
    indexStackSize = anIndexStackSize;
}

- (NSMutableArray *) lookahead
{
    return lookahead; 
}

- (void) setLookahead: (NSMutableArray *) aLookahead
{
    if (lookahead != aLookahead) {
        [aLookahead retain];
        [lookahead release];
        lookahead = aLookahead;
    }
}

- (void) addToLookahead: (id)lookaheadObject
{
    [[self lookahead] addObject: lookaheadObject];
}
- (void) removeFromLookahead: (id)lookaheadObject
{
    [[self lookahead] removeObject: lookaheadObject];
}


@end
