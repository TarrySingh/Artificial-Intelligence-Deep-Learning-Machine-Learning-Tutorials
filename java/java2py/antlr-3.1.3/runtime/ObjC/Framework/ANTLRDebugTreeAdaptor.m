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

#import "ANTLRDebugTreeAdaptor.h"


@implementation ANTLRDebugTreeAdaptor


- (id) initWithTreeAdaptor:(id<ANTLRTreeAdaptor>)aTreeAdaptor debugListener:(id<ANTLRDebugEventListener>)aDebugListener
{
	self = [super init];
	if (self) {
		[self setDebugListener:aDebugListener];
		[self setTreeAdaptor:aTreeAdaptor];
	}
	return self;
}

- (void) dealloc
{
    [self setDebugListener: nil];
    [self setTreeAdaptor: nil];
    [super dealloc];
}

- (id<ANTLRDebugEventListener>) debugListener
{
    return debugListener; 
}

- (void) setDebugListener: (id<ANTLRDebugEventListener>) aDebugListener
{
    if (debugListener != aDebugListener) {
        [(id<ANTLRTreeAdaptor,NSObject>)aDebugListener retain];
        [(id<ANTLRTreeAdaptor,NSObject>)debugListener release];
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

#pragma mark -
#pragma mark Proxy implementation

// anything else that hasn't some debugger event assicioated with it, is simply
// forwarded to the actual token stream
- (void) forwardInvocation:(NSInvocation *)anInvocation
{
	[anInvocation invokeWithTarget:[self treeAdaptor]];
}

#pragma mark -

#pragma mark Construction

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>) payload
{
	id<ANTLRTree> newTree = [treeAdaptor newTreeWithToken:payload];
	[debugListener createNode:[treeAdaptor uniqueIdForTree:newTree] fromTokenAtIndex:[payload tokenIndex]];
	return newTree;
}

- (id<ANTLRTree>) emptyTree
{
	id<ANTLRTree> newTree = [treeAdaptor newEmptyTree];
	[debugListener createNilNode:[treeAdaptor uniqueIdForTree:newTree]];
	return newTree;
}

/*	We don't have debug events for those:
 - (id) copyNode:(id<ANTLRTree>)aNode
{
}
- (id) copyTree:(id<ANTLRTree>)aTree
{
}
*/

- (void) addChild:(id<ANTLRTree>)child toTree:(id<ANTLRTree>)aTree
{
	[treeAdaptor addChild:child toTree:aTree];
	[debugListener addChild:[treeAdaptor uniqueIdForTree:child] toTree:[self uniqueIdForTree:aTree]];
}

- (id<ANTLRTree>) makeNode:(id<ANTLRTree>)newRoot parentOf:(id<ANTLRTree>)oldRoot
{
	id<ANTLRTree> newTree = [treeAdaptor makeNode:newRoot parentOf:oldRoot];
	[debugListener makeNode:[treeAdaptor uniqueIdForTree:newTree] parentOf:[self uniqueIdForTree:oldRoot]];
	return newTree;
}

/* handle by forwardInvocation: 
- (unsigned int) uniqueIdForTree:(id<ANTLRTree>)aNode
{
}
*/

#pragma mark Rewrite Rules

 - (void) addTokenAsChild:(id<ANTLRToken>)child toTree:(id<ANTLRTree>)aTree
{
	id<ANTLRTree> newChild = [self newTreeWithToken:child];
	[self addChild:newChild toTree:aTree];
}

- (id<ANTLRTree>) makeToken:(id<ANTLRToken>)newRoot parentOf:(id<ANTLRTree>)oldRoot
{
	id<ANTLRTree> newNode = [self newTreeWithToken:newRoot];
	return [self makeNode:newNode parentOf:oldRoot];
}

- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType
{
	id<ANTLRTree> newTree = [treeAdaptor newTreeWithTokenType:tokenType];
	[debugListener createNode:[treeAdaptor uniqueIdForTree:newTree] text:nil type:tokenType];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithTokenType:(int)tokenType text:(NSString *)tokenText
{
	id<ANTLRTree> newTree = [treeAdaptor newTreeWithTokenType:tokenType text:tokenText];
	[debugListener createNode:[treeAdaptor uniqueIdForTree:newTree] text:tokenText type:tokenType];
	return newTree;
}
- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType
{
	id<ANTLRTree> newTree = [treeAdaptor newTreeWithToken:fromToken tokenType:tokenType];
	[debugListener createNode:[treeAdaptor uniqueIdForTree:newTree] text:[fromToken text] type:tokenType];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken tokenType:(int)tokenType text:(NSString *)tokenText
{
	id<ANTLRTree> newTree = [treeAdaptor newTreeWithToken:fromToken tokenType:tokenType text:tokenText];
	[debugListener createNode:[treeAdaptor uniqueIdForTree:newTree] text:tokenText type:tokenType];
	return newTree;
}

- (id<ANTLRTree>) newTreeWithToken:(id<ANTLRToken>)fromToken text:(NSString *)tokenText
{
	id<ANTLRTree> newTree = [treeAdaptor newTreeWithToken:fromToken text:tokenText];
	[debugListener createNode:[treeAdaptor uniqueIdForTree:newTree] text:tokenText type:[fromToken type]];
	return newTree;
}

#pragma mark Content

/* handled by forwardInvocation:
- (int) tokenTypeForNode:(id<ANTLRTree>)aNode
{
}
- (void) setTokenType:(int)tokenType forNode:(id)aNode
{
}

- (NSString *) textForNode:(id<ANTLRTree>)aNode
{
}
- (void) setText:(NSString *)tokenText forNode:(id<ANTLRTree>)aNode
{
}
*/
- (void) setBoundariesForTree:(id<ANTLRTree>)aTree fromToken:(id<ANTLRToken>)startToken toToken:(id<ANTLRToken>)stopToken
{
	[treeAdaptor setBoundariesForTree:aTree fromToken:startToken toToken:stopToken];
	if (aTree && startToken && stopToken) {
		[debugListener setTokenBoundariesForTree:[aTree hash] start:[startToken tokenIndex] stop:[stopToken tokenIndex]];
	}
}
/* handled by forwardInvocation:
- (int) tokenStartIndexForTree:(id<ANTLRTree>)aTree
{
}
- (int) tokenStopIndexForTree:(id<ANTLRTree>)aTree
{
}
*/

#pragma mark Navigation / Tree Parsing
/* handled by forwardInvocation:
- (id<ANTLRTree>) childForNode:(id<ANTLRTree>) aNode atIndex:(int) i
{
}
- (int) childCountForTree:(id<ANTLRTree>) aTree
{
}
*/

@end
